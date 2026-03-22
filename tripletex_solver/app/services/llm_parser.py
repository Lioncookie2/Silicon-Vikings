import logging
import json
import os
from typing import Optional, Dict, Any, List

from app.models.structured_task import StructuredTask

logger = logging.getLogger(__name__)

class LLMParser:
    """
    Service for parsing natural language prompts into structured Task objects using an LLM.
    Currently defaults to Google Gemini using structured outputs.
    """
    def __init__(self):
        # We assume the user has set either GOOGLE_API_KEY or OPENAI_API_KEY as an env variable
        # Prøver å laste fra .env fil hvis den finnes
        from dotenv import load_dotenv
        load_dotenv()
        
        self.api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No LLM API key found in environment variables. Parsing may fail.")

    def parse_prompt(self, prompt: str, files: Optional[List[Dict[str, str]]] = None) -> StructuredTask:
        """
        Takes a raw natural language prompt and optional files, returns a strongly typed StructuredTask.
        Uses the newest Gemini 2.5 Flash model for the best performance and language support.
        """
        logger.info(f"Parsing prompt of length {len(prompt)} med {len(files) if files else 0} filer...")
        
        # If no API key is set, we return a fallback dummy response to prevent crashing
        if not self.api_key:
            logger.error("Cannot call LLM: No API key provided.")
            return self._fallback_parse(prompt)

        try:
            from google import genai
            from google.genai import types
            import base64

            client = genai.Client(api_key=self.api_key)
            
            # Klargjør innholdet til Gemini (Tekst + eventuelle PDF/Bilder)
            contents = []
            
            # Legg til filer hvis vi har dem
            if files:
                for f in files:
                    mime_type = f.get("mime_type", "")
                    content_base64 = f.get("content_base64", "")
                    if content_base64 and mime_type:
                        try:
                            raw_bytes = base64.b64decode(content_base64)
                            contents.append(
                                types.Part.from_bytes(data=raw_bytes, mime_type=mime_type)
                            )
                            logger.info(f"La til fil i LLM context: {mime_type}")
                        except Exception as e:
                            logger.error(f"Feil ved lasting av fil til LLM: {e}")
            
            # Legg til selve prompten og systeminstruksjonene
            prompt_text = f"""
                You are an expert accounting AI. Your job is to classify accounting tasks and extract structured data from user prompts and attached documents (PDFs, receipts, invoices).
                The prompt can be in one of 7 languages (nb, en, es, pt, nn, de, fr). You must handle all of them automatically.

                The output MUST be a valid JSON object conforming to the schema of the StructuredTask class.
                Extract exact names, emails, dates (convert to YYYY-MM-DD if needed), amounts, etc.
                
                TRIPLETEX API GUIDELINES / RECIPES:
                - CREATE_EMPLOYEE: Extract name, email, and if they should be an account administrator.
                - CREATE_CUSTOMER: Extract customer name and organization number if present.
                - CREATE_SUPPLIER: Extract supplier name and organization number if present.
                - CREATE_PRODUCT: Extract product name, price excluding VAT, product number (if any) and whether it is VAT free (isVatFree).
                - CREATE_PROJECT: Extract project name, customer name, and project manager.
                - CREATE_ORDER: Extract customer name, product name, quantity, unit price ex VAT, and order date.
                - CREATE_INVOICE: Same as order, but often implies converting an existing or implicit order to an invoice. Extract customer and amount. Set `sendInvoice=true` if it asks to send it. Set `isCreditNote=true` if reversing/crediting an existing invoice.
                - SEND_INVOICE: Task specifically asking to send an existing invoice to a customer.
                - REGISTER_INCOMING_PAYMENT: Customer paid an invoice. Extract customerName, amount, and org nr into `payment_data`.
                - REGISTER_SUPPLIER_INVOICE: Extract supplier name, amount, and description (e.g., rent, equipment) into `supplier_invoice_data`.
                - REGISTER_TRAVEL_EXPENSE: Extract employee name, trip title, and date into `travel_expense_data`.
                - REGISTER_EMPLOYMENT_AND_SALARY: Extract employee name, start date, and annual salary.
                - FIND_AND_FIX_VOUCHER_ERROR / DELETE_DUPLICATE_VOUCHER / ISSUE_REMINDER_FEE: Provide the detailed reasoning and context in `voucher_correction_data`.
                - COMPLEX_TASK: Use this task type if the prompt asks you to fix an existing accounting error, delete duplicates, correct VAT on existing vouchers, or do any multi-step data correction that requires searching the existing ledger/API to find the specific error first.
                
                IMPORTANT: If the user provides a document (PDF/Image), extract the relevant information from it (e.g. invoice amounts, supplier names, dates) and populate the appropriate data object.
                
                USER PROMPT:
                "{prompt}"
                """
            contents.append(prompt_text)
            
            # Using Gemini 2.5 Flash for best speed and capabilities
            response = client.models.generate_content(
                model='gemini-2.5-pro',
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=StructuredTask,
                    temperature=0.0, # Deterministic outputs
                ),
            )
            
            # The response text will be a valid JSON string matching the Pydantic schema
            data = json.loads(response.text)
            
            # Sørg for at empty extracted_entities er tillatt
            if 'extracted_entities' not in data:
                data['extracted_entities'] = None
                
            return StructuredTask(**data)
            
        except Exception as e:
            logger.error(f"Error calling LLM for prompt parsing: {str(e)}")
            # Fallback to a dumb parser if LLM fails
            return self._fallback_parse(prompt)
            
    def _fallback_parse(self, prompt: str) -> StructuredTask:
        """Simple rule-based fallback if the LLM fails or is unavailable."""
        lower_prompt = prompt.lower()
        
        task_type = "UNKNOWN"
        if "ansatt" in lower_prompt or "employee" in lower_prompt:
            task_type = "CREATE_EMPLOYEE"
        elif "kunde" in lower_prompt or "customer" in lower_prompt:
            task_type = "CREATE_CUSTOMER"
        elif "faktura" in lower_prompt or "invoice" in lower_prompt:
            task_type = "CREATE_INVOICE"
            
        return StructuredTask(
            task_type=task_type,
            confidence=0.1,
            language="unknown",
            reasoning="Fallback parser used because LLM failed or API key was missing.",
            extracted_entities=json.dumps({"raw_prompt": prompt})
        )
