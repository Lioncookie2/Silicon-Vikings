import requests
import json
import os
from dotenv import load_dotenv
from app.services.tripletex_client import TripletexClient

load_dotenv()
SESSION_TOKEN = "eyJ0b2tlbklkIjoyMTQ3NjkzMzIwLCJ0b2tlbiI6IjI0M2RkMjc1LWU2NDEtNDdkNy05ZDhiLTI2NWY4YzczNjYwZiJ9"
API_URL = "https://kkpqfuj-amager.tripletex.dev/v2"

client = TripletexClient(base_url=API_URL, session_token=SESSION_TOKEN)

try:
    print("Checking bank settings...")
    res = client.get("bank")
    print("GET bank:", res)
except Exception as e:
    print("Error:", e)
