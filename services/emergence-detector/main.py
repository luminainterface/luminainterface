from fastapi import FastAPI
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="emergence-detector", version="1.0.0")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "emergence-detector",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    return {"message": "emergence-detector service is running", "port": 8979}

if __name__ == "__main__":
    logger.info("Starting emergence-detector on port 8979")
    uvicorn.run(app, host="0.0.0.0", port=8979)
