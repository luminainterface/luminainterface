from fastapi import FastAPI
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="multi-agent-system", version="1.0.0")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "multi-agent-system",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    return {"message": "multi-agent-system service is running", "port": 8970}

if __name__ == "__main__":
    logger.info("Starting multi-agent-system on port 8970")
    uvicorn.run(app, host="0.0.0.0", port=8970)
