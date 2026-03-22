import requests
import json
from app.services.tripletex_client import TripletexClient

SESSION_TOKEN = "eyJ0b2tlbklkIjoyMTQ3NjkzMzIwLCJ0b2tlbiI6IjI0M2RkMjc1LWU2NDEtNDdkNy05ZDhiLTI2NWY4YzczNjYwZiJ9"
API_URL = "https://kkpqfuj-amager.tripletex.dev/v2"
client = TripletexClient(base_url=API_URL, session_token=SESSION_TOKEN)

print("Checking invoices for bank account error...")
try:
    s = client.search("invoice", {"fields": "*"})
    print(s)
except Exception as e:
    print(e)
    
print("\nCreating invoice directly with hardcoded account?")
try:
    # See if invoice can take bank account directly in payload
    # Just creating a minimal test
    s = client.post("invoice", {
        "invoiceDate": "2023-10-10",
        "invoiceDueDate": "2023-10-24",
        "customer": {"id": 12345}, # invalid but lets see the error
        "bankAccountNumber": "12345678903"
    })
    print(s)
except Exception as e:
    print(e)
