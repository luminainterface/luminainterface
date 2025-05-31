from fastapi import FastAPI
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="enhanced-crawler-nlp", version="1.0.0")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "enhanced-crawler-nlp",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    return {"message": "enhanced-crawler-nlp service is running", "port": 8850}

if __name__ == "__main__":
    logger.info("Starting enhanced-crawler-nlp on port 8850")
    uvicorn.run(app, host="0.0.0.0", port=8850)
