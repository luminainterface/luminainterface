from fastapi import FastAPI
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="transcript-ingest", version="1.0.0")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "transcript-ingest",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    return {"message": "transcript-ingest service is running", "port": 9264}

if __name__ == "__main__":
    logger.info("Starting transcript-ingest on port 9264")
    uvicorn.run(app, host="0.0.0.0", port=9264)
