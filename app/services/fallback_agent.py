import logging
import json
import os
from typing import Optional
from app.services.tripletex_client import TripletexClient
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

class FallbackAgent:
    """
    En AI-agent som tar over når den vanlige deterministiske koden kræsjer (f.eks ved 422 Valideringsfeil),
    eller hvis oppgaven er klassifisert som COMPLEX_TASK.
    Agenten har tilgang til å kalle Tripletex API-et direkte ved hjelp av Gemini Function Calling (Tools).
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY") # Vi bruker default Google hvis mulig
        self.genai_client = None
        if self.api_key:
            try:
                self.genai_client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Klarte ikke å initialisere genai.Client i FallbackAgent: {e}")
                
        # Last inn Swagger
        self.swagger_data = None
        swagger_path = os.path.join(os.path.dirname(__file__), "swagger.json")
        if os.path.exists(swagger_path):
            try:
                with open(swagger_path, "r", encoding="utf-8") as f:
                    self.swagger_data = json.load(f)
            except Exception as e:
                logger.error(f"Klarte ikke å laste inn swagger.json i agent: {e}")

    def _search_api_docs(self, keyword: str) -> str:
        """
        Søker i OpenAPI/Swagger dokumentasjonen for Tripletex v2.
        Returnerer relevante endepunkter, metoder og deres påkrevde JSON schemas.
        """
        if not self.swagger_data:
            return "Swagger dokumentasjon ikke tilgjengelig."
        
        keyword = keyword.lower()
        results = []
        paths = self.swagger_data.get("paths", {})
        
        for path, methods in paths.items():
            if keyword in path.lower() or any(keyword in str(op.get("tags", [])).lower() or keyword in op.get("summary", "").lower() for m, op in methods.items() if m in ["get", "post", "put", "delete"]):
                path_info = f"Path: {path}\n"
                for method, op in methods.items():
                    if method not in ["get", "post", "put", "delete"]: continue
                    summary = op.get("summary", "")
                    path_info += f"  - {method.upper()}: {summary}\n"
                    # Forenklet parameter/schema info
                    params = op.get("parameters", [])
                    body_param = next((p for p in params if p.get("in") == "body"), None)
                    if body_param:
                        schema_ref = body_param.get("schema", {}).get("$ref")
                        if schema_ref:
                            model_name = schema_ref.split("/")[-1]
                            path_info += f"    Payload Model: {model_name}\n"
                results.append(path_info)
        
        if not results:
            return f"Fant ingen dokumentasjon for '{keyword}'."
            
        output = "Funnet dokumentasjon:\n" + "\n".join(results[:15]) # Return max 15 to fit context
        return output

    def solve(self, client: TripletexClient, prompt: str, task_type: str, extracted_data: str, error_message: str = None) -> str:
        """
        Kjører agent-loopen for å prøve å løse feilen.
        """
        if not self.genai_client:
            raise Exception("Agent mangler API-nøkkel eller kunne ikke starte.")

        # Vi definerer API-verktøyet inni metoden slik at den har direkte tilgang til "client"
        def call_tripletex_api(method: str, endpoint: str, payload: dict = None) -> dict:
            """
            Kaller Tripletex v2 REST API.
            
            Args:
                method: GET, POST, PUT, eller DELETE
                endpoint: API-stien (f.eks. 'customer' eller 'ledger/account')
                payload: JSON dictionary for POST/PUT. Valgfritt.
                
            Returns:
                API-responsen i JSON, eller feilinformasjon.
            """
            try:
                logger.info(f"🤖 Agent kaller API: {method} {endpoint}")
                # Hent riktig HTTP-metode
                method_upper = method.upper()
                
                # Siden vi har bygget en robust TripletexClient wrapper, bruker vi den direkte
                # Vi bruker _request for å ha full kontroll
                kwargs = {}
                if method_upper == "GET":
                    kwargs["params"] = payload
                else:
                    kwargs["json"] = payload or {}
                    
                response = client._request(method_upper, endpoint, **kwargs)
                return {"success": True, "data": response}
            except Exception as e:
                logger.warning(f"🤖 Agent fikk feil fra API: {e}")
                return {"success": False, "error": str(e)}

        def search_tripletex_api_docs(keyword: str) -> str:
            """
            Søk i Tripletex API dokumentasjonen (Swagger/OpenAPI). 
            Bruk ALLTID dette verktøyet FØR du prøver å gjette payload-formatet på et ukjent endepunkt (f.eks. 'travelExpense', 'salary', 'invoice').
            Returnerer paths, HTTP metoder og navn på payload-modeller.
            
            Args:
                keyword: Søkeord, f.eks. "travelExpense", "employee", "project"
            """
            logger.info(f"🤖 Agent søker i API docs for: {keyword}")
            return self._search_api_docs(keyword)

        # Instruksjonen til Gemini
        system_instruction = """Du er en autonom AI-agent ("FallbackAgent") bygget inn i et Tripletex ERP integrasjonssystem.
Din jobb er å løse oppgaver som det vanlige systemet kræsjet på (f.eks på grunn av manglende forutsetninger, 422 Validation Error) eller heilt nye oppgaver vi mangler kode for.

Du har tilgang til TO verktøy: 
1. `call_tripletex_api`: For å utforske (GET), opprette (POST) og oppdatere (PUT) i Tripletex.
2. `search_tripletex_api_docs`: SØK I DOKUMENTASJONEN HVIS DU ER USIKKER PÅ ENDEPUNKT ELLER FELTNAVN!

PROSESS FOR Å LØSE UKJENTE OPPGAVER:
0. Hvis oppgaven er helt ukjent, bruk ALLTID `search_tripletex_api_docs` først med et nøkkelord (f.eks. "travel" eller "expense") for å finne riktig endpoint.
1. Les den opprinnelige oppgaven (Prompt) og hva som feilet.
2. Identifiser hvorfor det feilet. (f.eks: mangler avdeling? mangler bankkonto? eksisterer e-posten fra før?)
3. Bruk `call_tripletex_api` for å hente nødvendig info eller rette feilen. Du kan kalle det flere ganger etter hverandre!
4. Når du har rettet opp den underliggende feilen, GJENNOMFØR selve opprinnelige oppgaven ved å kalle riktig API.
5. Svar deretter brukeren med en VELDIG KORT og presis oppsummering av hva du gjorde for å løse oppgaven.

VIKTIG: Du MÅ fortsette å bruke verktøyene inntil oppgaven er faktisk løst. Tripletex-sandboxen er ofte helt tom, så forutsetninger (kunde, prosjekt, bankkonto, ansatt) må ofte opprettes dynamisk."""

        # Kontekst-meldingen vi sender inn for å starte prosessen
        context_msg = f"OPPRINNELIG OPPGAVE: {prompt}\n\nKLASSIFISERT SOM: {task_type}\n\nEKSTRAHERT DATA: {extracted_data}"
        if error_message:
            context_msg += f"\n\nFEILMELDING FRA FORRIGE KODE-FORSØK:\n{error_message}\n\nVennligst les feilmeldingen nøye, fiks problemet (bruk APIet), og deretter utfør den opprinnelige oppgaven i sin helhet."
        else:
            context_msg += "\n\nDette er en kompleks oppgave som krever steg-for-steg undersøkelser. Vennligst bruk API-verktøyet til å løse den."

        logger.info("🤖 Starter Agentic Fallback Loop...")
        
        try:
            # Start en chat session som støtter multi-turn tool calling
            chat = self.genai_client.chats.create(
                model="gemini-2.5-pro",
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.0, # Deterministic behavior
                    tools=[call_tripletex_api, search_tripletex_api_docs]
                )
            )
            
            # Gemini Python SDK håndterer automatisk funksjonskallene inntil agenten returnerer tekst
            response = chat.send_message(context_msg)
            
            logger.info("🤖 Agent fullførte oppgaven med svar.")
            return response.text
            
        except Exception as e:
            logger.error(f"Agenten feilet fatalt: {e}")
            raise Exception(f"Agent-fallback feilet: {e}")
