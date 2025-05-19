"""
Crawler service for the Lumina Core system
"""

import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Lumina Crawler")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/crawl")
async def crawl(url: str):
    """Crawl a given URL"""
    try:
        # TODO: Implement actual crawling logic
        return {"status": "success", "url": url}
    except Exception as e:
        logger.error(f"Error crawling {url}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8400"))
    uvicorn.run(app, host="0.0.0.0", port=port) 