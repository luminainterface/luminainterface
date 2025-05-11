import os
import json
import logging
import tempfile
from typing import Dict, List, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
from gtts import gTTS
import soundfile as sf
import numpy as np
import datetime
import threading
import time
import asyncio

# Environment variables
API_KEY = os.getenv("OUTPUT_ENGINE_API_KEY", "changeme")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
TEMP_DIR = os.getenv("TEMP_DIR", "/tmp/output-engine")
DEBUG_NO_OLLAMA = os.getenv("DEBUG_NO_OLLAMA", "0") == "1"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("output_engine")

app = FastAPI(title="Output Engine Service")

# Rate limiting
RATE_LIMIT_WINDOW = 60  # seconds
MAX_REQUESTS = 2  # max requests per minute
request_timestamps = []
rate_limit_lock = threading.Lock()

def check_rate_limit():
    global request_timestamps
    now = time.time()
    with rate_limit_lock:
        # Remove timestamps older than the window
        request_timestamps = [ts for ts in request_timestamps if now - ts < RATE_LIMIT_WINDOW]
        if len(request_timestamps) >= MAX_REQUESTS:
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")
        request_timestamps.append(now)

class AnalyzedConcept(BaseModel):
    term: str
    definition: str
    related_concepts: List[Dict[str, Any]]
    insights: List[str]
    narrative: str
    embedding: List[float]

class ConceptAnalysis(BaseModel):
    concepts: List[AnalyzedConcept]
    overall_narrative: str

class ProcessedOutput(BaseModel):
    text: str
    audio_path: str
    metadata: Dict[str, Any]

async def get_ollama_completion(prompt: str, max_retries: int = 2) -> str:
    if DEBUG_NO_OLLAMA:
        return "This is a stub narrative for testing."
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": "mistral",
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "num_predict": 5  # Minimize GPU usage
                        }
                    }
                )
                response.raise_for_status()
                return response.json()["response"]
        except Exception as e:
            logger.error(f"Failed to get completion from Ollama (attempt {attempt+1}): {e}")
            if attempt == max_retries - 1:
                raise HTTPException(status_code=500, detail="Failed to get completion from Ollama")
            await asyncio.sleep(1)

async def generate_natural_language(analysis: ConceptAnalysis) -> str:
    """Generate natural language output using Ollama."""
    prompt = f"""Create a natural, engaging narrative that explains these concepts and their relationships:

    Overall Context: {analysis.overall_narrative}

    Individual Concepts:
    {chr(10).join([f"- {concept.term}: {concept.narrative}" for concept in analysis.concepts])}

    Generate a coherent, flowing narrative that:
    1. Introduces the overall topic and its importance
    2. Explains each concept in a logical sequence
    3. Highlights key relationships and insights
    4. Concludes with a meaningful summary
    
    Make the text engaging and easy to understand while maintaining accuracy."""
    narrative = await get_ollama_completion(prompt)
    return narrative

def generate_speech(text: str, output_path: str) -> None:
    """Generate speech from text using gTTS."""
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(output_path)
        
        # Convert to 16-bit PCM WAV for better compatibility
        data, samplerate = sf.read(output_path)
        sf.write(output_path, data, samplerate, subtype='PCM_16')
    except Exception as e:
        logger.error(f"Failed to generate speech: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate speech")

@app.post("/process")
async def process_analysis(analysis: ConceptAnalysis) -> ProcessedOutput:
    """Process the analyzed concepts and generate natural language output."""
    check_rate_limit()
    try:
        # Ensure temp directory exists
        os.makedirs(TEMP_DIR, exist_ok=True)
        
        # Generate natural language narrative
        narrative = await generate_natural_language(analysis)
        
        # Generate speech
        audio_path = os.path.join(TEMP_DIR, "output.wav")
        generate_speech(narrative, audio_path)
        
        # Prepare metadata
        metadata = {
            "num_concepts": len(analysis.concepts),
            "concept_terms": [concept.term for concept in analysis.concepts],
            "generated_at": datetime.datetime.now().isoformat(),
            "model": "mistral",
            "audio_format": "WAV",
            "audio_sample_rate": 44100,
            "audio_channels": 1
        }
        
        return ProcessedOutput(
            text=narrative,
            audio_path=audio_path,
            metadata=metadata
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/")
async def root():
    logger.info("Root endpoint accessed.")
    return {"status": "Output Engine is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000) 