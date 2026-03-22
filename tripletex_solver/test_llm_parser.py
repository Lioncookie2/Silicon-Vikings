import asyncio
import os
from app.services.llm_parser import LLMParser

async def test_parser():
    # Sjekker at nøkkelen finnes
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Advarsel: Ingen GOOGLE_API_KEY funnet i miljøet.")
    else:
        print(f"Fant API-nøkkel: {api_key[:10]}...")

    parser = LLMParser()
    
    # Test 1: Norsk, opprette ansatt
    prompt1 = "Kan du opprette en ny ansatt som heter Kari Nordmann? E-posten hennes er kari.nordmann@example.com. Hun skal være kontoadministrator."
    print(f"\n--- Tester: '{prompt1}' ---")
    result1 = parser.parse_prompt(prompt1)
    print(result1.model_dump_json(indent=2))
    
    # Test 2: Spansk, opprette kunde
    prompt2 = "Por favor, crea un nuevo cliente llamado Empresa de Software SL. Son un cliente, no un proveedor."
    print(f"\n--- Tester: '{prompt2}' ---")
    result2 = parser.parse_prompt(prompt2)
    print(result2.model_dump_json(indent=2))

if __name__ == "__main__":
    asyncio.run(test_parser())
