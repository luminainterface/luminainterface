from fastapi import FastAPI
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="optimal-lora-router", version="1.0.0")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "optimal-lora-router",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    return {"message": "optimal-lora-router service is running", "port": 5030}

if __name__ == "__main__":
    logger.info("Starting optimal-lora-router on port 5030")
    uvicorn.run(app, host="0.0.0.0", port=5030)
