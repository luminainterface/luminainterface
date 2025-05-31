from fastapi import FastAPI
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="rag-graph", version="1.0.0")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "rag-graph",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    return {"message": "rag-graph service is running", "port": 8921}

if __name__ == "__main__":
    logger.info("Starting rag-graph on port 8921")
    uvicorn.run(app, host="0.0.0.0", port=8921)
