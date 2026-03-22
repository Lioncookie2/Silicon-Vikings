import requests
import json
from app.services.tripletex_client import TripletexClient

SESSION_TOKEN = "eyJ0b2tlbklkIjoyMTQ3NjkzMzIwLCJ0b2tlbiI6IjI0M2RkMjc1LWU2NDEtNDdkNy05ZDhiLTI2NWY4YzczNjYwZiJ9"
API_URL = "https://kkpqfuj-amager.tripletex.dev/v2"
client = TripletexClient(base_url=API_URL, session_token=SESSION_TOKEN)

print("Checking bank settings...")
try:
    accs = client.get("bank/account")
    print(accs)
except Exception as e:
    pass

print("\nChecking bank/settings...")
try:
    s = client.get("bank/settings")
    print(s)
except Exception as e:
    print(e)
    
print("\nChecking company...")
try:
    s = client.get("company")
    print(json.dumps(s, indent=2))
except Exception as e:
    print(e)
