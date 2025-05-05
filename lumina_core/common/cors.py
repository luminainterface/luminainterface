from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from typing import List, Optional

def add_cors(app: FastAPI, origins: Optional[List[str]] = None) -> None:
    """Attach permissive CORS middleware in dev or whitelist in prod.

    Args:
        app: FastAPI application instance.
        origins: List of allowed origins. Defaults to localhost Vite dev if not provided.
    """
    if origins is None:
        origins = ["http://localhost:5173", "http://127.0.0.1:5173"]

    # If * is included we can just allow all
    allow_all = "*" in origins

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if allow_all else origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    ) 