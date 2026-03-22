import requests
import json
from app.services.tripletex_client import TripletexClient

SESSION_TOKEN = "eyJ0b2tlbklkIjoyMTQ3NjkzMzIwLCJ0b2tlbiI6IjI0M2RkMjc1LWU2NDEtNDdkNy05ZDhiLTI2NWY4YzczNjYwZiJ9"
API_URL = "https://kkpqfuj-amager.tripletex.dev/v2"
client = TripletexClient(base_url=API_URL, session_token=SESSION_TOKEN)

print("Checking company/details?")
try:
    c = client.get("company/settings")
    print("SETTINGS:", c)
except Exception as e:
    print(e)
    
try:
    c = client.get("company/1") # ID is fake but maybe we get fields
    print("COMPANY 1:", c)
except Exception as e:
    print(e)
