import json
import os
from app.services.tripletex_client import TripletexClient
from app.services.executor import TaskExecutor
from app.models.structured_task import StructuredTask

# Mock token from the active terminal / run_test_request.py
# If run_test_request.py is running, it uses the credentials there.
# Let's read run_test_request.py to get the token.
