from fastapi import FastAPI, WebSocket, HTTPException
from lumina_core.common.cors import add_cors

app = FastAPI(title="Event Mux")
add_cors(app) 