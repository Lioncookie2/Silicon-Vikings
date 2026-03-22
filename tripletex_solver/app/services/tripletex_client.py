import requests
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class TripletexClient:
    """
    Wrapper for Tripletex v2 REST API via the AI competition proxy.
    """
    def __init__(self, base_url: str, session_token: str):
        # AI competition proxy format: username "0", password is the token
        self.base_url = base_url.rstrip('/')
        self.auth = ("0", session_token)
        self.session = requests.Session()
        self.session.auth = self.auth
        
        # Timeout settings - keep it reasonable to fail fast and retry
        self.timeout = 10
        self.call_log = []

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        # Ensure we always get json back
        headers = kwargs.get('headers', {})
        if 'Accept' not in headers:
            headers['Accept'] = 'application/json'
        kwargs['headers'] = headers
            
        logger.info(f"API {method} {endpoint}")
        
        call_entry = {
            "method": method,
            "endpoint": endpoint,
            "params": kwargs.get("params", {}),
            "payload": kwargs.get("json", {}),
            "status": "pending",
            "response": None,
            "error": None
        }
        self.call_log.append(call_entry)
        
        try:
            response = self.session.request(method, url, timeout=self.timeout, **kwargs)
            call_entry["status"] = response.status_code
            
            # Raise exception for 4xx and 5xx errors
            if not response.ok:
                error_msg = f"HTTP {response.status_code} på {method} {endpoint}"
                try:
                    error_details = response.json()
                    call_entry["error"] = error_details
                    if "validationMessages" in error_details:
                        msgs = [f"{m.get('field', 'Generell')}: {m.get('message')}" for m in error_details["validationMessages"]]
                        error_msg += f". Valideringsfeil: {', '.join(msgs)}"
                    elif "message" in error_details:
                        error_msg += f". Melding: {error_details['message']}"
                except:
                    call_entry["error"] = response.text
                    error_msg += f". Raw body: {response.text}"
                    
                logger.error(error_msg)
                raise Exception(error_msg)
            
            # Return JSON if possible, otherwise empty dict
            if response.text:
                try:
                    data = response.json()
                    call_entry["response"] = data
                    return data
                except:
                    call_entry["response"] = response.text
            
            call_entry["response"] = {}
            return {}
            
        except requests.exceptions.HTTPError as e:
            call_entry["error"] = f"HTTP Error: {e.response.status_code}"
            logger.error(f"HTTP Error på {method} {url}: {e.response.status_code}")
            logger.error(f"Response body: {e.response.text}")
            raise Exception(f"HTTP {e.response.status_code}: {e.response.text}")
        except Exception as e:
            if call_entry["error"] is None:
                call_entry["error"] = str(e)
            logger.error(f"Request Error on {method} {url}: {str(e)}")
            raise

    def get_call_log(self) -> List[Dict]:
        return self.call_log

    # --- GET Methods ---
    def get(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        return self._request("GET", endpoint, params=params)

    def search(self, endpoint: str, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Utility for fetching a list of values from Tripletex (handles 'values' key)"""
        if params is None:
            params = {}
            
        # Default to fetching all reasonable fields if not specified
        if "fields" not in params:
            params["fields"] = "*"
            
        response = self.get(endpoint, params=params)
        return response.get("values", [])

    def get_by_id(self, endpoint: str, id: int, fields: str = "*") -> Dict[str, Any]:
        """Fetch a specific entity by ID"""
        response = self.get(f"{endpoint}/{id}", params={"fields": fields})
        return response.get("value", {})

    # --- POST/PUT/DELETE Methods ---
    def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        response = self._request("POST", endpoint, json=data)
        return response.get("value", response)

    def put(self, endpoint: str, id: Any, data: Dict[str, Any] = None) -> Dict[str, Any]:
        url_end = f"{endpoint}/{id}" if id is not None else endpoint
        response = self._request("PUT", url_end, json=data or {})
        return response.get("value", response)

    def delete(self, endpoint: str, id: int) -> None:
        self._request("DELETE", f"{endpoint}/{id}")

    # --- Domain Specific Helpers (Examples to build upon) ---
    def create_employee(self, first_name: str, last_name: str, email: str, **kwargs) -> Dict[str, Any]:
        data = {
            "firstName": first_name,
            "lastName": last_name,
            "email": email,
            **kwargs
        }
        return self.post("employee", data)
        
    def create_customer(self, name: str, is_customer: bool = True, **kwargs) -> Dict[str, Any]:
        data = {
            "name": name,
            "isCustomer": is_customer,
            **kwargs
        }
        return self.post("customer", data)
