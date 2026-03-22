from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

class TripletexCredentials(BaseModel):
    base_url: str = Field(description="Proxy API URL for the submission")
    session_token: str = Field(description="Session token for authentication")

class SolveRequest(BaseModel):
    prompt: str
    files: Optional[List[Dict[str, str]]] = Field(default_factory=list, description="List of base64 encoded files or URLs")
    tripletex_credentials: TripletexCredentials

class FileInfo(BaseModel):
    filename: str
    mime_type: str
    has_content: bool

class SolveResponse(BaseModel):
    status: str
    task_type: Optional[str] = None
    files_received: Optional[List[FileInfo]] = None
    execution_time_ms: Optional[int] = None
    details: Optional[Dict[str, Any]] = None
