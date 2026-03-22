from fastapi import APIRouter
from fastapi.responses import HTMLResponse
import json

try:
    from google.cloud import logging
    logging_client = logging.Client()
except ImportError:
    logging_client = None

router = APIRouter(tags=["dashboard"])

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tripletex Agent Live Logs</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .severity-INFO { border-left-color: #3b82f6; } /* Blue */
        .severity-WARNING { border-left-color: #f59e0b; } /* Yellow */
        .severity-ERROR { border-left-color: #ef4444; } /* Red */
    </style>
</head>
<body class="bg-gray-50 text-gray-800 font-sans h-screen flex flex-col">
    <!-- Header -->
    <header class="bg-white border-b shadow-sm sticky top-0 z-10">
        <div class="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8 flex justify-between items-center">
            <h1 class="text-xl font-bold text-gray-900 flex items-center gap-2">
                <svg class="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
                Tripletex Agent Live Logs
            </h1>
            <div class="flex items-center space-x-4">
                <div class="flex space-x-2 text-sm font-medium">
                    <label class="flex items-center space-x-1 cursor-pointer">
                        <input type="checkbox" id="filter-info" checked class="rounded text-blue-600 focus:ring-blue-500">
                        <span>INFO</span>
                    </label>
                    <label class="flex items-center space-x-1 cursor-pointer">
                        <input type="checkbox" id="filter-warn" checked class="rounded text-yellow-500 focus:ring-yellow-500">
                        <span>WARNING</span>
                    </label>
                    <label class="flex items-center space-x-1 cursor-pointer">
                        <input type="checkbox" id="filter-error" checked class="rounded text-red-600 focus:ring-red-500">
                        <span>ERROR</span>
                    </label>
                </div>
                <div class="text-sm text-gray-500" id="status-text">Connecting...</div>
            </div>
        </div>
    </header>

    <!-- Main Content -->
    <main class="flex-1 overflow-y-auto p-4 sm:p-6 lg:p-8 bg-gray-50">
        <div class="max-w-7xl mx-auto space-y-4" id="log-container">
            <!-- Cards will be injected here -->
        </div>
        <div id="loading" class="text-center py-10 text-gray-400 hidden">
            <svg class="animate-spin h-8 w-8 mx-auto mb-2" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Fetching latest logs...
        </div>
    </main>

    <script>
        const logContainer = document.getElementById('log-container');
        const statusText = document.getElementById('status-text');
        const loadingIndicator = document.getElementById('loading');
        
        const filters = {
            INFO: document.getElementById('filter-info'),
            WARNING: document.getElementById('filter-warn'),
            ERROR: document.getElementById('filter-error'),
            DEBUG: document.getElementById('filter-info') // group debug with info
        };

        // Re-render when filters change
        Object.values(filters).forEach(cb => {
            cb.addEventListener('change', () => renderLogs(lastLogs));
        });

        let lastLogs = [];
        let isFirstLoad = true;

        async function fetchLogs() {
            try {
                if (isFirstLoad) loadingIndicator.classList.remove('hidden');
                statusText.textContent = 'Updating...';
                
                const response = await fetch('/api/logs');
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                
                const data = await response.json();
                lastLogs = data.logs || [];
                renderLogs(lastLogs);
                
                statusText.textContent = `Last updated: ${new Date().toLocaleTimeString('en-US', { hour12: false })}`;
                statusText.classList.remove('text-red-500');
            } catch (err) {
                console.error('Failed to fetch logs:', err);
                statusText.textContent = 'Connection error';
                statusText.classList.add('text-red-500');
            } finally {
                loadingIndicator.classList.add('hidden');
                isFirstLoad = false;
            }
        }

        function formatTimestamp(isoString) {
            if (!isoString) return 'Unknown Time';
            // Keeping UTC as requested
            const d = new Date(isoString);
            return d.toISOString().replace('T', ' ').substring(0, 19) + ' UTC';
        }

        function renderLogs(logs) {
            logContainer.innerHTML = '';
            
            const filteredLogs = logs.filter(log => {
                const sev = log.severity || 'INFO';
                if (sev === 'INFO' || sev === 'DEBUG') return filters.INFO.checked;
                if (sev === 'WARNING') return filters.WARNING.checked;
                if (sev === 'ERROR' || sev === 'CRITICAL') return filters.ERROR.checked;
                return true;
            });

            if (filteredLogs.length === 0) {
                logContainer.innerHTML = '<div class="text-center py-10 text-gray-400 bg-white rounded-lg border border-dashed">No logs match the current filters.</div>';
                return;
            }

            filteredLogs.forEach(log => {
                const severity = log.severity || 'INFO';
                const msg = log.message || 'No message';
                const ts = formatTimestamp(log.timestamp);
                
                const card = document.createElement('div');
                card.className = `bg-white shadow-sm border rounded-lg p-4 border-l-4 severity-${severity} transition-colors hover:bg-gray-50`;
                
                // Build a rich content area
                let badges = '';
                if (log.payload?.method && log.payload?.path) {
                    const methodColor = log.payload.method === 'GET' ? 'bg-green-100 text-green-800' : 'bg-purple-100 text-purple-800';
                    badges += `<span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${methodColor} mr-2">${log.payload.method} ${log.payload.path}</span>`;
                }
                if (log.payload?.status) {
                    const statusColor = log.payload.status >= 400 ? 'bg-red-100 text-red-800' : 'bg-blue-100 text-blue-800';
                    badges += `<span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${statusColor} mr-2">HTTP ${log.payload.status}</span>`;
                }
                if (log.payload?.step !== undefined) {
                    badges += `<span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-800 mr-2">Step ${log.payload.step}</span>`;
                }
                if (log.payload?.request_id) {
                    const shortRid = log.payload.request_id.substring(0, 8);
                    badges += `<span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-500 mr-2" title="${log.payload.request_id}">rid:${shortRid}</span>`;
                }

                let detailsHtml = '';
                if (log.payload?.detail) {
                    detailsHtml += `<div class="mt-2 text-sm text-red-600 bg-red-50 p-2 rounded border border-red-100 whitespace-pre-wrap font-mono">${log.payload.detail}</div>`;
                }
                if (log.payload?.reasoning) {
                    detailsHtml += `<div class="mt-2 text-sm text-gray-600 bg-gray-50 p-2 rounded border border-gray-100 italic">🤖 "${log.payload.reasoning}"</div>`;
                }

                card.innerHTML = `
                    <div class="flex justify-between items-start">
                        <div class="flex-1 min-w-0">
                            <div class="flex items-center mb-1">
                                <span class="text-xs text-gray-500 font-mono mr-3">${ts}</span>
                                <span class="text-xs font-bold ${severity === 'ERROR' ? 'text-red-600' : severity === 'WARNING' ? 'text-yellow-600' : 'text-blue-600'} mr-3">${severity}</span>
                                ${badges}
                            </div>
                            <p class="text-sm font-medium text-gray-900 break-words">${msg}</p>
                            ${detailsHtml}
                        </div>
                    </div>
                `;
                logContainer.appendChild(card);
            });
        }

        // Fetch logs every 5 seconds
        fetchLogs();
        setInterval(fetchLogs, 5000);
    </script>
</body>
</html>
"""

@router.get("/logs", response_class=HTMLResponse)
async def dashboard_ui():
    """Serves the frontend HTML for the real-time logs dashboard."""
    return HTML_CONTENT

@router.get("/api/logs")
async def fetch_logs():
    """Fetches the latest 100 logs from Google Cloud Logging for this service."""
    if not logging_client:
        return {"error": "google-cloud-logging is not installed or configured correctly", "logs": []}
    
    try:
        # Filter for Cloud Run logs from this specific service that contain our structured payload
        filter_str = (
            'resource.type="cloud_run_revision" '
            'AND resource.labels.service_name="tripletex-agent" '
            'AND jsonPayload.log_schema="v2-rich"'
        )
        
        # We fetch up to 100 entries, sorted descending by timestamp
        entries = logging_client.list_entries(
            filter_=filter_str,
            order_by=logging.DESCENDING,
            max_results=100
        )
        
        logs = []
        for entry in entries:
            # GCP logging entries have a json_payload if they were logged as structured JSON
            payload = entry.payload if isinstance(entry.payload, dict) else {}
            
            # Extract standard fields
            log_item = {
                "timestamp": entry.timestamp.isoformat() if entry.timestamp else None,
                "severity": entry.severity,
                "message": payload.get("message", str(entry.payload) if not isinstance(entry.payload, dict) else "No message"),
                "payload": payload
            }
            logs.append(log_item)
            
        return {"logs": logs}
    except Exception as e:
        return {"error": str(e), "logs": []}
