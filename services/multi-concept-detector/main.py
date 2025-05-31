from fastapi import FastAPI
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-Concept Detector", version="1.0.0")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "multi-concept-detector",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    return {"message": "Multi-Concept Detector service is running", "port": 8860}

if __name__ == "__main__":
    logger.info("Starting Multi-Concept Detector on port 8860")
    uvicorn.run(app, host="0.0.0.0", port=8860) 