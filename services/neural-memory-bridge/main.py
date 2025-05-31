from fastapi import FastAPI
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Neural Memory Bridge", version="1.0.0")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "neural-memory-bridge",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    return {"message": "Neural Memory Bridge service is running", "port": 8892}

if __name__ == "__main__":
    logger.info("Starting Neural Memory Bridge on port 8892")
    uvicorn.run(app, host="0.0.0.0", port=8892) 