from fastapi import FastAPI
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="consensus-manager", version="1.0.0")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "consensus-manager",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    return {"message": "consensus-manager service is running", "port": 8978}

if __name__ == "__main__":
    logger.info("Starting consensus-manager on port 8978")
    uvicorn.run(app, host="0.0.0.0", port=8978)
