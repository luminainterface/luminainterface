from fastapi import FastAPI
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="vector-store", version="1.0.0")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "vector-store",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    return {"message": "vector-store service is running", "port": 9262}

if __name__ == "__main__":
    logger.info("Starting vector-store on port 9262")
    uvicorn.run(app, host="0.0.0.0", port=9262)
