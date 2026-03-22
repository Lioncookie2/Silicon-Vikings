"""Deterministic order creation via LLM extraction."""
from __future__ import annotations

import json
import os
from typing import Any

from pydantic import BaseModel, Field

from ..structured_log import log_event
from ..tripletex_client import TripletexClient


class OrderLineExtractor(BaseModel):
    product_number: str | None = Field(default=None, description="The product number, e.g. '7898'")
    product_name: str | None = Field(default=None, description="The name of the product")
    count: float = Field(default=1.0, description="The quantity/count")
    unit_price: float | None = Field(default=None, description="The price per unit excluding VAT")


class OrderExtractor(BaseModel):
    customer_name_hint: str = Field(description="The name of the customer the order is for")
    order_date: str | None = Field(default=None, description="The order date in YYYY-MM-DD")
    lines: list[OrderLineExtractor] = Field(description="The lines on the order")
    create_invoice: bool = Field(default=False, description="True if the task also asks to convert the order to an invoice or invoice it")


def _extract_order_data(prompt: str, today: str) -> OrderExtractor | None:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)
        model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

        resp = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=f"Extract order details. Today is {today}.",
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=OrderExtractor,
            ),
        )
        if resp.text:
            data = json.loads(resp.text)
            return OrderExtractor.model_validate(data)
    except Exception as e:
        log_event("WARNING", "order_llm_extract_failed", error=str(e))
        pass

    return None


def _resolve_customer_id(client: TripletexClient, name: str) -> int | None:
    r = client.get("/customer", params={"name": name, "fields": "id", "count": 1})
    if r.status_code == 200:
        vals = r.json().get("values") or []
        if vals:
            return vals[0].get("id")
    return None


def _resolve_product_id(client: TripletexClient, number: str | None, name: str | None) -> int | None:
    if not number and not name:
        return None
        
    # Try by number first
    if number:
        r = client.get("/product", params={"number": number, "fields": "id", "count": 1})
        if r.status_code == 200 and r.json().get("values"):
            return r.json()["values"][0]["id"]
            
    # Try by name
    if name:
        r = client.get("/product", params={"name": name, "fields": "id", "count": 1})
        if r.status_code == 200 and r.json().get("values"):
            return r.json()["values"][0]["id"]
            
    return None


def handle_create_order(prompt: str, client: TripletexClient, today: str) -> bool:
    """End-to-end extraction and API execution for creating an order."""
    extracted = _extract_order_data(prompt, today)
    if not extracted:
        log_event("WARNING", "order_handler_extraction_null")
        return False

    cust_id = _resolve_customer_id(client, extracted.customer_name_hint)
    if not cust_id:
        log_event("WARNING", "order_handler_missing_customer", name=extracted.customer_name_hint)
        return False

    # 1. Create the parent order
    body: dict[str, Any] = {
        "customer": {"id": cust_id},
        "orderDate": extracted.order_date or today,
    }

    resp = client.post("/order", json=body)
    if resp.status_code not in (200, 201):
        log_event("WARNING", "order_handler_post_failed", status=resp.status_code, text=resp.text[:200])
        return False

    order_id = resp.json().get("value", {}).get("id")
    if not order_id:
        return False

    log_event("INFO", "order_handler_created_parent", id=order_id)

    # 2. Add the order lines
    success = True
    for line in extracted.lines:
        line_body: dict[str, Any] = {
            "order": {"id": order_id},
            "count": line.count,
        }
        
        prod_id = _resolve_product_id(client, line.product_number, line.product_name)
        if prod_id:
            line_body["product"] = {"id": prod_id}
        elif line.product_name:
            line_body["description"] = line.product_name
        else:
            line_body["description"] = "Varelinje"
            
        if line.unit_price is not None:
            line_body["unitPriceExcludingVatCurrency"] = line.unit_price

        lr = client.post("/order/orderline", json=line_body)
        if lr.status_code not in (200, 201):
            log_event("WARNING", "order_handler_line_failed", status=lr.status_code, text=lr.text[:200])
            success = False

    # 3. Optionally create invoice from order
    if success and extracted.create_invoice:
        inv_body = {
            "invoiceDate": today,
            "customer": {"id": cust_id},
            "orders": [{"id": order_id}]
        }
        ir = client.post("/invoice", json=inv_body)
        if ir.status_code in (200, 201):
            log_event("INFO", "order_handler_created_invoice", id=ir.json().get("value", {}).get("id"))
        else:
            log_event("WARNING", "order_handler_invoice_failed", status=ir.status_code, text=ir.text[:200])
            # Even if invoice failed, the order was created
            return False

    return success
