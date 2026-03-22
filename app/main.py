import logging
import os
import base64
import json
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from app.models.schemas import SolveRequest, SolveResponse, FileInfo
from app.services.llm_parser import LLMParser
from app.services.tripletex_client import TripletexClient
from app.services.executor import TaskExecutor
from app.services.fallback_agent import FallbackAgent

# Sett opp logg-mappe
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Basic logging setup med fil-lagring for at du kan se tilbake på gamle feil
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Tripletex Solver API")
llm_parser = LLMParser()
fallback_agent = FallbackAgent()

TEMP_FILE_DIR = Path("temp_files")
TEMP_FILE_DIR.mkdir(exist_ok=True)

import time

# Minne-database for det visuelle dashboardet
task_history = []

@app.post("/solve", response_model=SolveResponse)
async def solve_task(request: SolveRequest):
    start_time = time.time()
    logger.info(f"Mottok oppgave. Prompt: '{request.prompt[:50]}...'")
    
    # Opprett en record for dashboardet
    task_record = {
        "id": len(task_history) + 1,
        "time": datetime.now().strftime("%H:%M:%S"),
        "prompt": request.prompt,
        "files": [],
        "task_type": "Analyserer...",
        "reasoning": "Venter på LLM...",
        "extracted_data": "",
        "status": "Processing",
        "result": "",
        "execution_time_ms": 0,
        "api_calls": []
    }
    task_history.insert(0, task_record) # Legg til øverst
    
    files_info = []
    if request.files:
        for f in request.files:
            filename = f.get("filename", "ukjent.fil")
            mime_type = f.get("mime_type", "unknown")
            content = f.get("content_base64", "")
            files_info.append(FileInfo(filename=filename, mime_type=mime_type, has_content=bool(content)))
            task_record["files"].append(filename)
            
            if content:
                try:
                    file_path = TEMP_FILE_DIR / filename
                    file_path.write_bytes(base64.b64decode(content))
                except Exception as e:
                    logger.error(f"Feil ved fillagring: {e}")
    
    try:
        # 1. & 2. LLM Parser med filstøtte
        # Cloud Run injiserer secrets via env. Sjekker eksplisitt.
        if "LLM_API_KEY" in os.environ:
            if not llm_parser.api_key:
                llm_parser.api_key = os.environ["LLM_API_KEY"]
            if getattr(fallback_agent, "genai_client", None) is None:
                from google import genai
                fallback_agent.api_key = os.environ["LLM_API_KEY"]
                fallback_agent.genai_client = genai.Client(api_key=fallback_agent.api_key)
            
        structured_task = llm_parser.parse_prompt(request.prompt, request.files)
        
        # Oppdater dashboardet med AI-ens tanker
        task_record["task_type"] = structured_task.task_type
        task_record["reasoning"] = structured_task.reasoning
        task_record["extracted_data"] = structured_task.model_dump_json(indent=2)
        
        # 3. Tripletex klient
        creds = request.tripletex_credentials.dict() if hasattr(request.tripletex_credentials, 'dict') else request.tripletex_credentials
        base_url = creds.get('base_url', 'https://kkpqfuj-amager.tripletex.dev/v2')
        session_token = creds.get('session_token') or creds.get('consumer_token')
        client = TripletexClient(base_url=base_url, session_token=session_token)

        # 4. & 5. Executor
        response_details = {}
        
        executor = TaskExecutor(client)
        try:
            status, msg = executor.execute(structured_task)
            response_details = {"message": msg, "llm_reasoning": structured_task.reasoning}
            task_record["status"] = status
            task_record["result"] = msg
            task_record["api_calls"] = client.get_call_log()
        except Exception as e:
            logger.warning(f"Happy Path feilet (eller mangler logikk) for {structured_task.task_type}: {e}. Starter Agentic Fallback...")
            task_record["status"] = "Agent Processing"
            try:
                # Vi sender feilmeldingen til agenten slik at den kan lese den og fikse det
                msg = fallback_agent.solve(client, request.prompt, structured_task.task_type, structured_task.model_dump_json(), str(e))
                response_details = {"message": msg, "llm_reasoning": f"Agent tok over etter feil: {str(e)}"}
                task_record["status"] = "Success"
                task_record["result"] = f"Agent fikset feilen/oppgaven: {msg}"
                task_record["api_calls"] = client.get_call_log()
            except Exception as agent_e:
                task_record["api_calls"] = client.get_call_log()
                raise Exception(f"Første forsøk feilet: {e}. Fallback-Agent feilet også: {agent_e}")

        execution_time_ms = int((time.time() - start_time) * 1000)
        task_record["execution_time_ms"] = execution_time_ms

        # Lagre vellykket eller advarsel til loggfil for feilsøking og evals
        task_filename = f"task_{task_record['id']}_{structured_task.task_type}_{int(time.time())}.json"
        
        # Vi lagrer full logg i tasks.jsonl, pluss en dedikert fil for denne oppgaven
        with open(LOG_DIR / "tasks.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(task_record, ensure_ascii=False) + "\n")
            
        with open(LOG_DIR / "evals" / task_filename, "w", encoding="utf-8") as f:
            f.write(json.dumps(task_record, indent=2, ensure_ascii=False))

        return SolveResponse(
            status="completed",
            task_type=structured_task.task_type,
            files_received=files_info,
            execution_time_ms=execution_time_ms,
            details=response_details
        )
        
    except Exception as e:
        error_msg = f"En feil oppstod: {str(e)}"
        logger.error(error_msg)
        task_record["status"] = "Error"
        task_record["result"] = error_msg
        
        execution_time_ms = int((time.time() - start_time) * 1000)
        task_record["execution_time_ms"] = execution_time_ms
        
        if 'client' in locals() and hasattr(client, 'get_call_log'):
            task_record["api_calls"] = client.get_call_log()
        else:
            if "api_calls" not in task_record:
                task_record["api_calls"] = []
        
        # Lagre feil til loggfil for feilsøking
        task_filename = f"error_task_{task_record['id']}_{int(time.time())}.json"
        
        with open(LOG_DIR / "tasks.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(task_record, ensure_ascii=False) + "\n")
            
        with open(LOG_DIR / "evals" / task_filename, "w", encoding="utf-8") as f:
            f.write(json.dumps(task_record, indent=2, ensure_ascii=False))
        
        return SolveResponse(
            status="completed",
            execution_time_ms=execution_time_ms,
            details={"error": error_msg}
        )

@app.get("/health")
def health_check():
    return {"status": "ok"}

# --- VISUELT DASHBOARD ---
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    html_content = """
    <!DOCTYPE html>
    <html lang="no">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Tripletex AI Solver Dashboard</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <meta http-equiv="refresh" content="5"> <!-- Oppdaterer siden hvert 5. sekund -->
        <script>
            function copyError(button, text) {
                navigator.clipboard.writeText(text).then(() => {
                    const originalText = button.innerText;
                    button.innerText = "Kopiert!";
                    button.classList.replace("bg-red-200", "bg-green-200");
                    button.classList.replace("text-red-700", "text-green-800");
                    setTimeout(() => {
                        button.innerText = originalText;
                        button.classList.replace("bg-green-200", "bg-red-200");
                        button.classList.replace("text-green-800", "text-red-700");
                    }, 2000);
                });
            }
        </script>
    </head>
    <body class="bg-gray-100 text-gray-800 font-sans p-6">
        <div class="max-w-6xl mx-auto">
            <h1 class="text-3xl font-bold mb-2 text-blue-600">🧠 Tripletex AI Solver Dashboard</h1>
            <p class="mb-6 text-gray-600">Her ser du live hva AI-en tenker og gjør når den mottar oppgaver.</p>
            
            <div class="space-y-6">
    """
    
    if not task_history:
        html_content += """
        <div class="bg-white p-8 rounded-lg shadow text-center text-gray-500">
            Ingen oppgaver mottatt enda. Kjør <code>python run_test_request.py</code> for å teste!
        </div>
        """
        
    for task in task_history:
        status_color = "bg-yellow-100 border-yellow-400 text-yellow-800"
        if task["status"] == "Success":
            status_color = "bg-green-100 border-green-400 text-green-800"
        elif task["status"] == "Error":
            status_color = "bg-red-100 border-red-400 text-red-800"
            
        files_badge = ""
        if task["files"]:
            files_badge = f"""<span class="ml-2 px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full border border-blue-200">📎 {', '.join(task['files'])}</span>"""
            
        html_content += f"""
        <div class="bg-white rounded-lg shadow-md overflow-hidden border border-gray-200">
            <div class="flex justify-between items-center px-4 py-3 border-b border-gray-200 bg-gray-50">
                <div class="font-semibold flex items-center">
                    <span class="text-gray-500 mr-2">#{task["id"]}</span> 
                    <span class="text-blue-700">{task["task_type"]}</span>
                    {files_badge}
                </div>
                <div class="flex items-center space-x-4">
                    <span class="text-xs font-mono bg-gray-200 text-gray-700 px-2 py-1 rounded">⏱️ {task.get("execution_time_ms", 0)} ms</span>
                    <span class="text-xs font-mono bg-purple-200 text-purple-800 px-2 py-1 rounded">🔌 {len(task.get("api_calls", []))} API calls</span>
                    <div class="text-sm text-gray-500">{task["time"]}</div>
                </div>
            </div>
            
            <div class="p-4 grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                    <h3 class="text-xs font-bold text-gray-400 uppercase tracking-wider mb-1">1. Oppgave (Prompt)</h3>
                    <p class="text-sm bg-gray-50 p-3 rounded border border-gray-100 italic">"{task["prompt"]}"</p>
                    
                    <h3 class="text-xs font-bold text-gray-400 uppercase tracking-wider mt-4 mb-1">2. AI-ens Tankeprosess (Reasoning)</h3>
                    <p class="text-sm text-gray-700 bg-indigo-50 p-3 rounded border border-indigo-100">💡 {task["reasoning"]}</p>
                </div>
                
                <div>
                    <h3 class="text-xs font-bold text-gray-400 uppercase tracking-wider mb-1">3. Strukturert JSON Data</h3>
                    <pre class="text-xs bg-gray-800 text-green-400 p-3 rounded overflow-x-auto h-32">{task["extracted_data"]}</pre>
                </div>
            </div>
            
            """
        if task.get("api_calls"):
            html_content += """
            <div class="px-4 py-3 border-t border-gray-200 bg-gray-50">
                <h3 class="text-xs font-bold text-gray-500 uppercase tracking-wider mb-2">4. API Trace</h3>
                <div class="space-y-2">
            """
            for call in task["api_calls"]:
                method_color = "bg-blue-100 text-blue-800"
                if call["method"] == "POST": method_color = "bg-green-100 text-green-800"
                elif call["method"] == "PUT": method_color = "bg-orange-100 text-orange-800"
                elif call["method"] == "DELETE": method_color = "bg-red-100 text-red-800"
                
                status_badge = f'<span class="text-xs font-mono px-2 py-1 rounded bg-gray-200">{call["status"]}</span>'
                if str(call["status"]).startswith("2"):
                    status_badge = f'<span class="text-xs font-mono px-2 py-1 rounded bg-green-200 text-green-800">{call["status"]}</span>'
                elif str(call["status"]).startswith("4") or str(call["status"]).startswith("5"):
                    status_badge = f'<span class="text-xs font-mono px-2 py-1 rounded bg-red-200 text-red-800">{call["status"]}</span>'
                    
                payload_str = ""
                if call["payload"]:
                    payload_str = f'<div class="mt-1 text-[10px] text-gray-500 font-mono break-all border-l-2 border-gray-300 pl-2">Payload: {json.dumps(call["payload"], ensure_ascii=False)}</div>'
                
                error_str = ""
                if call["error"]:
                    err_content = json.dumps(call["error"], ensure_ascii=False) if isinstance(call["error"], dict) else call["error"]
                    error_str = f'<div class="mt-1 text-[10px] text-red-500 font-mono break-all border-l-2 border-red-300 pl-2">Error: {err_content}</div>'
                
                html_content += f"""
                    <div class="bg-white p-2 rounded border border-gray-200 shadow-sm flex flex-col">
                        <div class="flex items-center justify-between">
                            <div class="flex items-center space-x-2">
                                <span class="{method_color} text-[10px] font-bold px-2 py-0.5 rounded">{call["method"]}</span>
                                <span class="font-mono text-xs text-gray-700">/{call["endpoint"]}</span>
                            </div>
                            {status_badge}
                        </div>
                        {payload_str}
                        {error_str}
                    </div>
                """
            html_content += """
                </div>
            </div>
            """
            
        html_content += f"""
            <div class="px-4 py-3 {status_color} border-t flex justify-between items-center">
                <div>
                    <span class="font-bold">{task["status"]}:</span> {task["result"]}
                </div>
                """
        
        if task["status"] == "Error":
            safe_error = task["result"].replace("'", "&#39;").replace('"', '&quot;')
            html_content += f"""<button onclick="copyError(this, '{safe_error}')" class="px-3 py-1 bg-red-200 hover:bg-red-300 text-red-700 font-medium text-xs rounded-md shadow-sm transition-colors">Kopier Feilmelding</button>"""
            
        html_content += """
            </div>
        </div>
        """
        
    html_content += """
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
