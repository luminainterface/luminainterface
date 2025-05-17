import os

# Service URLs
GRAPH_API_URL = os.getenv("GRAPH_API_URL", "http://graph-api:8200")
# Use :8828 for local development, :8000 for Docker Compose
CONCEPT_DICT_URL = os.getenv("CONCEPT_DICT_URL", "http://localhost:8828") 