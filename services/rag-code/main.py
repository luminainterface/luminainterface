from fastapi import FastAPI
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="rag-code", version="1.0.0")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "rag-code",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    return {"message": "rag-code service is running", "port": 8922}

if __name__ == "__main__":
    logger.info("Starting rag-code on port 8922")
    uvicorn.run(app, host="0.0.0.0", port=8922)
