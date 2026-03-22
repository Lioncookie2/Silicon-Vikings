"""Deterministic accounting dimension handler."""
from __future__ import annotations

import json
import os
from typing import Any

from pydantic import BaseModel, Field

from ..structured_log import log_event
from ..tripletex_client import TripletexClient

class DimensionExtractor(BaseModel):
    dimension_name: str = Field(description="The name of the custom dimension to create, e.g. 'Kostsenter'")
    dimension_values: list[str] = Field(description="The values for this dimension, e.g. ['Kundeservice', 'Økonomi']")
    post_voucher: bool = Field(description="True if we should also post a voucher with these dimensions")

def _extract_dimension_data(prompt: str) -> DimensionExtractor | None:
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
                system_instruction="Extract custom accounting dimension details.",
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=DimensionExtractor,
            ),
        )
        if resp.text:
            data = json.loads(resp.text)
            return DimensionExtractor.model_validate(data)
    except Exception as e:
        log_event("WARNING", "dimension_llm_extract_failed", error=str(e))
        pass

    return None

def handle_dimensions(prompt: str, client: TripletexClient) -> bool:
    extracted = _extract_dimension_data(prompt)
    if not extracted:
        return False
        
    # 1. Create or find Dimension Name
    r = client.get("/ledger/accountingDimensionName", params={"fields": "id,name,dimensionName,dimensionIndex", "count": 10})
    if r.status_code != 200:
        return False
        
    dim_name_id = None
    for dim in r.json().get("values", []):
        if dim.get("dimensionName", "").lower() == extracted.dimension_name.lower():
            dim_name_id = dim["id"]
            break
            
    if not dim_name_id:
        body = {"name": extracted.dimension_name, "dimensionName": extracted.dimension_name}
        r2 = client.post("/ledger/accountingDimensionName", json=body)
        if r2.status_code in (200, 201):
            dim_name_id = r2.json().get("value", {}).get("id")
        else:
            log_event("WARNING", "dimension_name_create_failed", status=r2.status_code)
            return False

    if not dim_name_id:
        return False
        
    # 2. Create the values
    for val in extracted.dimension_values:
        body = {
            "name": val,
            "accountingDimensionName": {"id": dim_name_id}
        }
        client.post("/ledger/accountingDimensionValue", json=body)
        
    log_event("INFO", "dimension_handler_success", dimension_name=extracted.dimension_name)
    
    # If the task requires posting a voucher, we let the LLM handle that part or we'd need a more complex extractor.
    if extracted.post_voucher:
        # Fall back to ReAct loop to post the voucher since we created the dimensions
        return False

    return True