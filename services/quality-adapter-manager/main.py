from fastapi import FastAPI
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="quality-adapter-manager", version="1.0.0")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "quality-adapter-manager",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    return {"message": "quality-adapter-manager service is running", "port": 8996}

if __name__ == "__main__":
    logger.info("Starting quality-adapter-manager on port 8996")
    uvicorn.run(app, host="0.0.0.0", port=8996)
