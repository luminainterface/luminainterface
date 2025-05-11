import os
import httpx
from fastapi import FastAPI, Request, HTTPException
import logging
import json
import asyncio
from pathlib import Path
from lumina_core.common.adapter_watcher import watch_and_reload

app = FastAPI()

MISTRAL_URL = os.getenv("MISTRAL_URL", "http://ollama:11436")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral")
ADAPTER_DIR = Path("/app/adapters")
ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("masterchat-llm")
logging.basicConfig(level=logging.INFO)

def load_adapter(path: str) -> None:
    """Load adapter into the model.
    
    For Ollama models, we use the Ollama API to load the adapter.
    For local models, we would load directly into the model.
    """
    try:
        # For Ollama models, we use the API to load the adapter
        if MISTRAL_URL.startswith("http://ollama"):
            # Ollama models don't support hot-swapping adapters
            logger.info("Ollama models don't support hot-swapping adapters")
            return
            
        # For local models (e.g., llama.cpp), we would load directly
        logger.info(f"Loading adapter from {path}")
        # TODO: Implement local model adapter loading
        # model.load_adapter(path)
        
    except Exception as e:
        logger.error(f"Failed to load adapter: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Start adapter watcher on startup"""
    asyncio.create_task(
        watch_and_reload(
            load_fn=load_adapter,
            adapter_dir=ADAPTER_DIR,
            service_name="masterchat-llm"
        )
    )

@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{MISTRAL_URL}/api/tags", timeout=5.0)
            resp.raise_for_status()
            return {"status": "healthy", "ollama_connected": True}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "ollama_connected": False, "error": str(e)}

@app.post("/chat")
async def chat_llm(payload: dict):
    prompt = payload.get("prompt", "")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{MISTRAL_URL}/api/generate",
                json={"prompt": prompt, "model": MISTRAL_MODEL},
                timeout=30.0
            )
            resp.raise_for_status()
            
            # Handle streaming response
            full_response = ""
            for line in resp.text.split('\n'):
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                    if chunk.get("response"):
                        full_response += chunk["response"]
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse chunk: {e}")
                    continue
            
            logger.info(f"Full LLM response: {full_response}")
            return {
                "response": full_response,
                "confidence": 0.8  # Default confidence for now
            }
    except Exception as e:
        logger.error(f"Mistral call failed: {e}")
        raise HTTPException(status_code=500, detail=f"Mistral call failed: {str(e)}")

@app.post("/adapters/{adapter_id}/reload")
async def reload_adapter(adapter_id: str):
    """Manually trigger adapter reload"""
    try:
        adapter_path = ADAPTER_DIR / f"{adapter_id}.bin"
        if not adapter_path.exists():
            raise HTTPException(status_code=404, detail=f"Adapter {adapter_id} not found")
            
        load_adapter(str(adapter_path))
        return {"status": "success", "message": f"Adapter {adapter_id} reloaded"}
    except Exception as e:
        logger.error(f"Failed to reload adapter: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 