import os
from dotenv import load_dotenv

load_dotenv()
import requests
import json

# DIN SANDBOX-INFO HER
# 1. Lim inn hele den lange session-tokenen fra skjermbildet ditt
SESSION_TOKEN = "eyJ0b2tlbklkIjoyMTQ3NjkzMzIwLCJ0b2tlbiI6IjI0M2RkMjc1LWU2NDEtNDdkNy05ZDhiLTI2NWY4YzczNjYwZiJ9"

# 2. Dette er URL-en fra bildet (beholdes som den er med mindre bildet endrer seg)
API_URL = "https://kkpqfuj-amager.tripletex.dev/v2"

# Prøv en faktura for å sjekke prosjekt, kunde, ansatt, ordrelinje-kompleks
payload = {
    "prompt": "Registrer 7 timar for Liv Brekke (liv.brekke@example.org) på aktiviteten 'Design' i prosjektet 'Datamigrering' for Strandvik AS (org.nr 962818684). Timesats: 1650 kr/t. Generer ein prosjektfaktura til kunden basert på dei registrerte timane.",
    "tripletex_credentials": {
        "base_url": API_URL,
        "session_token": SESSION_TOKEN
    }
}

try:
    print("Sender forespørsel til lokal solver (http://127.0.0.1:8080/solve)...")
    response = requests.post("http://127.0.0.1:8080/solve", json=payload, timeout=60)
    print(f"\nStatus: {response.status_code}")
    print("Svar:")
    print(json.dumps(response.json(), indent=2))
except requests.exceptions.ConnectionError:
    print("FEIL: Fikk ikke kontakt med serveren. Kjører den? (uvicorn app.main:app --reload --port 8080)")
except Exception as e:
    print(f"Error: {e}")
