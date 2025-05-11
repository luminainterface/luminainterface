import os
import httpx
import logging
import numpy as np
import asyncio
import traceback
from typing import Dict, List, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import time

# Environment variables
CONCEPT_DICT_URL = os.getenv("CONCEPT_DICT_URL", "http://concept-dictionary:8000")
OUTPUT_ENGINE_URL = os.getenv("OUTPUT_ENGINE_URL", "http://output-engine:9000")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
API_KEY = os.getenv("ANALYZER_API_KEY", "changeme")
DEBUG_NO_OLLAMA = os.getenv("DEBUG_NO_OLLAMA", "0") == "1"

# Setup logging with more detail
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("data_analyzer")
logger.setLevel(logging.DEBUG)

app = FastAPI(title="Data Analyzer Service")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
RATE_LIMIT_WINDOW = 60  # 1 minute window
MAX_REQUESTS = 2  # Reduced to 2 requests per minute to prevent GPU overload
request_timestamps = []

class RateLimitExceeded(Exception):
    pass

def check_rate_limit():
    """Check if the current request exceeds rate limits."""
    global request_timestamps
    current_time = time.time()
    
    # Remove timestamps older than the window
    request_timestamps = [ts for ts in request_timestamps if current_time - ts < RATE_LIMIT_WINDOW]
    
    if len(request_timestamps) >= MAX_REQUESTS:
        raise RateLimitExceeded("Rate limit exceeded. Please try again later.")
    
    request_timestamps.append(current_time)

class AnalyzedConcept(BaseModel):
    term: str
    definition: str
    insights: List[str]
    narrative: str
    related_concepts: List[Dict[str, Any]] = []
    embedding: List[float] = []

class ConceptAnalysis(BaseModel):
    concepts: List[AnalyzedConcept]
    overall_narrative: str

class ConceptRequest(BaseModel):
    term: str
    definition: str

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests and their duration."""
    start_time = time.time()
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        logger.info(f"Request to {request.url.path} completed in {duration:.2f}s with status {response.status_code}")
        return response
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Request to {request.url.path} failed after {duration:.2f}s: {str(e)}\n{traceback.format_exc()}")
        raise

async def get_ollama_completion(prompt: str, max_retries: int = 3) -> str:
    print("[DEBUG] Entered get_ollama_completion")
    if DEBUG_NO_OLLAMA:
        # Return a stubbed response for testing
        return "INSIGHTS:\n- This is a stub insight 1\n- This is a stub insight 2\n- This is a stub insight 3\n\nNARRATIVE:\nThis is a stub narrative for testing."
    for attempt in range(max_retries):
        try:
            logger.debug(f"Attempting Ollama completion (attempt {attempt + 1}/{max_retries})")
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": "mistral",
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "num_predict": 5  # Minimize GPU usage
                        }
                    }
                )
                response.raise_for_status()
                result = response.json()["response"]
                print(f"[DEBUG] Raw Ollama response (from get_ollama_completion): {result}")
                logger.debug(f"Raw Ollama response: {result}")
                logger.debug(f"Ollama completion successful (attempt {attempt + 1})")
                return result
        except Exception as e:
            logger.error(f"Ollama completion failed (attempt {attempt + 1}): {str(e)}")
            if attempt == max_retries - 1:
                raise HTTPException(status_code=500, detail=f"Failed to get completion after {max_retries} attempts: {str(e)}")
            await asyncio.sleep(1)  # Wait before retry

async def send_to_output_engine(analysis: ConceptAnalysis) -> bool:
    """Send analyzed data to the output engine."""
    try:
        logger.debug("Sending analysis to output engine")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{OUTPUT_ENGINE_URL}/process",
                json=analysis.dict(),
                headers={"X-API-Key": API_KEY}
            )
            response.raise_for_status()
            logger.debug("Successfully sent to output engine")
            return True
    except Exception as e:
        logger.error(f"Failed to send to output engine: {str(e)}")
        return False

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    return JSONResponse(
        status_code=429,
        content={"detail": str(exc)}
    )

@app.post("/analyze")
async def analyze_concept(request: ConceptRequest) -> Dict[str, Any]:
    """Analyze a single concept and generate insights."""
    try:
        logger.info(f"Received analysis request for concept: {request.term}")
        
        # Check rate limit
        check_rate_limit()
        
        # Create a single concept for analysis
        concept = {
            "term": request.term,
            "definition": request.definition
        }
        
        # Generate insights and narrative in a single Ollama call
        prompt = f"""You are a precise and structured AI assistant. Your task is to analyze concepts and provide insights and a narrative in a very specific format.

        SYSTEM: You MUST follow the exact format below. Do not add any other text, sections, or explanations. The response must start with "INSIGHTS:" and contain exactly 3 insights, followed by "NARRATIVE:" and the narrative.

        Concept: {concept['term']}
        Definition: {concept['definition']}
        
        RESPONSE FORMAT (copy this exactly):
        INSIGHTS:
        - [First insight]
        - [Second insight]
        - [Third insight]
        
        NARRATIVE:
        [Your narrative here]

        CRITICAL FORMAT RULES:
        1. Start with "INSIGHTS:" (exactly as shown)
        2. List exactly 3 insights, each starting with "- " (exactly as shown)
        3. Then write "NARRATIVE:" (exactly as shown)
        4. Write your narrative after that
        5. Do not add any other sections, text, or explanations
        6. Do not add any markdown formatting
        7. Do not add any line breaks between insights
        8. Do not add any line breaks before or after NARRATIVE:"""
        
        print("[DEBUG] Requesting Ollama completion...")
        logger.debug("Requesting Ollama completion")
        response_text = await get_ollama_completion(prompt)
        print(f"[DEBUG] Raw Ollama response: {response_text}")
        logger.debug(f"Raw Ollama response (from analyze_concept): {response_text}")
        
        # Parse the response with more robust error handling
        try:
            # Split on NARRATIVE: and ensure we have exactly two parts
            parts = response_text.split("NARRATIVE:")
            if len(parts) != 2:
                raise ValueError("Response missing NARRATIVE section")
                
            # Extract and clean insights
            insights_part = parts[0].replace("INSIGHTS:", "").strip()
            insights = []
            for line in insights_part.split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    insights.append(line[2:].strip())
            
            if len(insights) != 3:
                raise ValueError(f"Expected exactly 3 insights, got {len(insights)}")
            
            # Extract and clean narrative
            narrative = parts[1].strip()
            if not narrative:
                raise ValueError("Empty narrative in response")
            
            logger.debug(f"Successfully parsed response with {len(insights)} insights")
            
            # Create analyzed concept
            analyzed_concept = AnalyzedConcept(
                term=concept["term"],
                definition=concept["definition"],
                insights=insights,
                narrative=narrative,
                related_concepts=[],
                embedding=[]
            )
            
            # Create analysis with single concept
            analysis = ConceptAnalysis(
                concepts=[analyzed_concept],
                overall_narrative=narrative
            )
            
            # Send to output engine
            success = await send_to_output_engine(analysis)
            if not success:
                raise HTTPException(status_code=500, detail="Failed to send to output engine")
            
            logger.info(f"Successfully analyzed concept: {request.term}")
            return {
                "status": "success",
                "message": "Concept analyzed and sent to output engine",
                "analysis": analysis.dict()
            }
        except ValueError as e:
            logger.error(f"Invalid response format: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    except RateLimitExceeded as e:
        logger.warning(f"Rate limit exceeded: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8500, log_level="debug") 