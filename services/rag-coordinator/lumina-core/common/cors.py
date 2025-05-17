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
        origins = ["*"]  # Allow all origins by default in dev

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    ) 