import os
from dotenv import load_dotenv
import requests
import json
from datetime import datetime

load_dotenv()

SESSION_TOKEN = "eyJ0b2tlbklkIjoyMTQ3NjkzMzIwLCJ0b2tlbiI6IjI0M2RkMjc1LWU2NDEtNDdkNy05ZDhiLTI2NWY4YzczNjYwZiJ9"
API_URL = "https://kkpqfuj-amager.tripletex.dev/v2"
auth = ("0", SESSION_TOKEN)

swagger = requests.get("https://tripletex.no/v2/swagger.json").json()
print("Entitlements:")
resp = requests.get(f"{API_URL}/employee/entitlement", auth=auth)
print(json.dumps(resp.json(), indent=2))
