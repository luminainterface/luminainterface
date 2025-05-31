from fastapi import FastAPI
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="concept-training-worker", version="1.0.0")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "concept-training-worker",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    return {"message": "concept-training-worker service is running", "port": 8851}

if __name__ == "__main__":
    logger.info("Starting concept-training-worker on port 8851")
    uvicorn.run(app, host="0.0.0.0", port=8851)
