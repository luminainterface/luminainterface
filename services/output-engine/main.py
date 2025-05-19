import os
import json
import logging
import tempfile
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx
from gtts import gTTS
import soundfile as sf
import numpy as np
import datetime
import threading
import time
import asyncio
from qdrant_client import QdrantClient
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from output_types import GeneratedOutput, ConceptPayload, RewardUpdate, TrainerVectorBatch
import redis
import torch
from transformers import AutoTokenizer
from optimum.intel import OVModelForCausalLM
from openvino.runtime import Core

# Environment variables
API_KEY = os.getenv("OUTPUT_ENGINE_API_KEY", "changeme")
TEMP_DIR = os.getenv("TEMP_DIR", "/tmp/output-engine")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
REDIS_URL = os.getenv("REDIS_URL", "redis://:02211998@redis:6379")
CONCEPT_DICT_URL = os.getenv("CONCEPT_DICT_URL", "http://concept-dictionary:8828")
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/phi-2")
DEVICE = os.getenv("DEVICE", "CPU")  # OpenVINO uses CPU, GPU, or NPU
USE_NPU = os.getenv("USE_NPU", "0") == "1"

# Initialize OpenVINO
logger.info("Initializing OpenVINO...")
ov_core = Core()
available_devices = ov_core.available_devices
logger.info(f"Available OpenVINO devices: {available_devices}")

# Select device based on availability and preference
if USE_NPU and "NPU" in available_devices:
    DEVICE = "NPU"
    logger.info("Using Intel NPU for inference")
elif "GPU" in available_devices:
    DEVICE = "GPU"
    logger.info("Using Intel GPU for inference")
else:
    DEVICE = "CPU"
    logger.info("Using CPU for inference")

# Initialize model and tokenizer
logger.info(f"Loading Phi-2 model on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

# Load model with OpenVINO optimization
model = OVModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    export=True,  # Export to OpenVINO format
    device=DEVICE,
    trust_remote_code=True
)

# Initialize clients
qdrant_client = QdrantClient(url=QDRANT_URL)
redis_client = redis.Redis.from_url(REDIS_URL)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("output_engine")

# Initialize FastAPI app
app = FastAPI(title="Output Engine Service")
Instrumentator().instrument(app).expose(app)

# Metrics
OUTPUT_GENERATED = Counter(
    'output_engine_generated_total',
    'Number of outputs generated',
    ['confidence_level']  # 'high', 'medium', 'low'
)

REWARD_UPDATES = Counter(
    'output_engine_reward_updates_total',
    'Number of reward updates',
    ['reason']
)

PROCESSING_LATENCY = Histogram(
    'output_engine_processing_latency_seconds',
    'Time spent processing outputs'
)

# Rate limiting
RATE_LIMIT_WINDOW = 60  # seconds
MAX_REQUESTS = 30  # max requests per minute (increased from 2 to 30)
request_timestamps = []
rate_limit_lock = threading.Lock()

def check_rate_limit():
    global request_timestamps
    now = time.time()
    with rate_limit_lock:
        request_timestamps = [ts for ts in request_timestamps if now - ts < RATE_LIMIT_WINDOW]
        if len(request_timestamps) >= MAX_REQUESTS:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
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

async def update_concept_reward(term: str, increment: int = 1, reason: Optional[str] = None) -> bool:
    """Update reward counter for a concept in Qdrant."""
    try:
        # Get current payload
        point = qdrant_client.retrieve(
            collection_name="concepts",
            ids=[term],
            with_payload=True
        )
        if not point:
            logger.warning(f"Concept {term} not found in Qdrant")
            return False

        # Update reward in payload
        payload = point[0].payload
        current_reward = payload.get("reward", 0)
        new_reward = current_reward + increment

        # Update in Qdrant
        qdrant_client.set_payload(
            collection_name="concepts",
            payload={
                "reward": new_reward,
                "last_reward_update": datetime.datetime.utcnow().isoformat(),
                "reward_reason": reason
            },
            points=[term]
        )

        REWARD_UPDATES.labels(reason=reason or "unknown").inc()
        logger.info(f"Updated reward for {term}: {current_reward} -> {new_reward}")
        return True
    except Exception as e:
        logger.error(f"Error updating reward for {term}: {e}")
        return False

async def get_model_completion(prompt: str, max_retries: int = 2) -> str:
    """Get completion from Phi-2 model with OpenVINO optimization."""
    start_time = time.time()
    for attempt in range(max_retries):
        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # Generate with OpenVINO optimized model
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Calculate confidence based on response length and token probabilities
            # Note: OpenVINO doesn't provide direct access to logits, so we use a simpler confidence metric
            confidence = min(1.0, max(0.0, len(response) / 1000))  # Simple length-based confidence
            
            # Log confidence level
            if confidence >= 0.8:
                OUTPUT_GENERATED.labels(confidence_level="high").inc()
            elif confidence >= 0.5:
                OUTPUT_GENERATED.labels(confidence_level="medium").inc()
            else:
                OUTPUT_GENERATED.labels(confidence_level="low").inc()
            
            return response.replace(prompt, "").strip()
            
        except Exception as e:
            logger.error(f"Failed to get completion (attempt {attempt+1}): {e}")
            if attempt == max_retries - 1:
                raise HTTPException(status_code=500, detail="Failed to get completion")
            await asyncio.sleep(1)
    return ""

async def generate_natural_language(analysis: ConceptAnalysis) -> tuple[str, float]:
    """Generate natural language output with confidence score."""
    prompt = f"""Create a natural, engaging narrative that explains these concepts and their relationships:

    Overall Context: {analysis.overall_narrative}

    Individual Concepts:
    {chr(10).join([f"- {concept.term}: {concept.narrative}" for concept in analysis.concepts])}

    Generate a coherent, flowing narrative that:
    1. Introduces the overall topic and its importance
    2. Explains each concept in a logical sequence
    3. Highlights key relationships and insights
    4. Concludes with a meaningful summary
    
    Make the text engaging and easy to understand while maintaining accuracy.
    
    Narrative:"""
    
    narrative = await get_model_completion(prompt)
    # Calculate confidence based on narrative length and content
    confidence = min(1.0, max(0.0, len(narrative) / 1000))  # Simple heuristic
    return narrative, confidence

def generate_speech(text: str, output_path: str) -> None:
    """Generate speech from text using gTTS."""
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(output_path)
        
        # Convert to 16-bit PCM WAV
        data, samplerate = sf.read(output_path)
        sf.write(output_path, data, samplerate, subtype='PCM_16')
    except Exception as e:
        logger.error(f"Failed to generate speech: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate speech")

@app.post("/process")
async def process_analysis(analysis: ConceptAnalysis) -> GeneratedOutput:
    """Process analyzed concepts and generate output with metrics."""
    check_rate_limit()
    start_time = time.time()
    
    try:
        os.makedirs(TEMP_DIR, exist_ok=True)
        
        # Generate narrative with confidence
        narrative, confidence = await generate_natural_language(analysis)
        
        # Generate speech
        audio_path = os.path.join(TEMP_DIR, f"output_{int(time.time())}.wav")
        generate_speech(narrative, audio_path)
        
        # Calculate processing time
        turn_ms = int((time.time() - start_time) * 1000)
        
        # Create excerpt (first 100 chars)
        excerpt = narrative[:100] + "..." if len(narrative) > 100 else narrative
        
        # Prepare metadata
        metadata = {
            "num_concepts": len(analysis.concepts),
            "concept_terms": [concept.term for concept in analysis.concepts],
            "model": "phi-2",
            "audio_format": "WAV",
            "audio_sample_rate": 44100,
            "audio_channels": 1,
            "confidence": confidence
        }
        
        output = GeneratedOutput(
            text=narrative,
            audio_path=audio_path,
            turn_ms=turn_ms,
            excerpt=excerpt,
            confidence=confidence,
            metadata=metadata,
            generated_at=datetime.datetime.utcnow(),
            reward=0
        )
        
        # Record processing latency
        PROCESSING_LATENCY.observe(time.time() - start_time)
        
        return output
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def add_concept_to_dictionary(term: str, definition: str, narrative: str) -> bool:
    """Add a concept to the Concept Dictionary via Redis stream."""
    try:
        # Create concept message
        concept_data = {
            "term": term,
            "definition": definition,
            "narrative": narrative,
            "metadata": {
                "source": "output-engine",
                "timestamp": datetime.datetime.utcnow().isoformat()
            }
        }
        
        # Serialize the concept data to JSON
        serialized_data = json.dumps(concept_data)
        
        # Publish to Redis stream
        stream_key = "concept:new"
        redis_client.xadd(
            stream_key,
            {"data": serialized_data},  # Redis expects string values
            maxlen=10000  # Keep last 10k messages
        )
        
        logger.info(f"Successfully published concept '{term}' to Redis stream {stream_key}")
        return True
    except Exception as e:
        logger.error(f"Failed to publish concept '{term}' to Redis stream: {e}")
        return False

@app.post("/respond")
async def respond(request: Dict[str, Any]) -> Dict[str, Any]:
    """Endpoint for dual-chat-router. Accepts a simple query payload and returns a compatible response."""
    # Accept both query and prompt fields for backward compatibility
    query = request.get("query", request.get("prompt", ""))
    session_id = request.get("session_id", None)

    # Extract potential concept term from query
    concept_term = query.strip().lower()

    # Build a dummy AnalyzedConcept (using a single concept) for ConceptAnalysis
    dummy_concept = AnalyzedConcept(
        term=concept_term,
        definition=f"Definition of {concept_term}",
        related_concepts=[],
        insights=[],
        narrative=query,
        embedding=[0.0] * 384  # dummy embedding (e.g. 384 floats)
    )
    dummy_analysis = ConceptAnalysis(concepts=[dummy_concept], overall_narrative=query)

    # Call the existing /process endpoint logic (generate narrative and speech)
    output = await process_analysis(dummy_analysis)

    # Add concept to dictionary (via Redis stream)
    await add_concept_to_dictionary(
        term=concept_term,
        definition=dummy_concept.definition,
        narrative=output.text
    )

    # Wait for the Concept Dictionary to process and store the concept
    concept_dict_url = os.getenv("CONCEPT_DICT_URL", "http://concept-dictionary:8828")
    concept_url = f"{concept_dict_url}/concepts/{concept_term.replace(' ', '%20')}"
    headers = {"X-API-Key": API_KEY}
    timeout = 30  # seconds
    poll_interval = 1  # second
    found = False
    concept_data = None
    for _ in range(timeout):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(concept_url, headers=headers)
                if resp.status_code == 200:
                    concept_data = resp.json()
                    found = True
                    break
        except Exception as e:
            logger.info(f"Waiting for concept in Concept Dictionary: {e}")
        await asyncio.sleep(poll_interval)
    if not found:
        raise HTTPException(status_code=504, detail=f"Concept '{concept_term}' not found in Concept Dictionary after {timeout} seconds.")

    # Return the enriched concept data from the Concept Dictionary
    return {
        "response": output.text,
        "confidence": output.confidence,
        "concepts_used": [concept_term],
        "concept_data": concept_data
    }

@app.post("/reward/{term}")
async def increment_reward(term: str, update: RewardUpdate) -> Dict[str, Any]:
    """Increment reward counter for a concept."""
    try:
        success = await update_concept_reward(
            term=term,
            increment=update.increment,
            reason=update.reason
        )
        if not success:
            raise HTTPException(status_code=404, detail="Concept not found")
        
        return {
            "status": "success",
            "term": term,
            "increment": update.increment,
            "reason": update.reason,
            "timestamp": update.timestamp.isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error incrementing reward: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint with dependency status."""
    try:
        # Check Qdrant connection
        qdrant_client.get_collections()
        qdrant_status = "healthy"
        
        # Check OpenVINO status
        ov_status = "healthy"
        if not model.is_loaded():
            ov_status = "unhealthy"
            logger.error("OpenVINO model not loaded")
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        qdrant_status = "unhealthy"
        ov_status = "unhealthy"
    
    return {
        "status": "healthy",
        "dependencies": {
            "qdrant": qdrant_status,
            "openvino": ov_status,
            "device": DEVICE
        },
        "timestamp": datetime.datetime.utcnow().isoformat()
    }

@app.middleware("http")
async def api_key_auth(request: Request, call_next):
    """API key authentication middleware."""
    if request.url.path.startswith("/health"):
        return await call_next(request)
    
    logger.info(f"Received headers: {request.headers}")
    logger.info(f"Received query params: {request.query_params}")
    
    # Check for API key in headers or query parameters
    api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
    logger.info(f"Received API key: {api_key}")
    logger.info(f"Expected API key: {API_KEY}")
    
    if api_key != API_KEY:
        logger.error(f"API key mismatch. Received: {api_key}, Expected: {API_KEY}")
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid or missing API key"}
        )
    return await call_next(request)

class OutputRequest(BaseModel):
    model_id: str
    query: str
    parameters: Dict[str, Any] = {}

class OutputResponse(BaseModel):
    prediction_id: str
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any]

@app.post("/output")
async def generate_output(request: OutputRequest):
    """Generate output for a given request."""
    check_rate_limit()
    start_time = time.time()
    
    try:
        prediction_id = f"{request.model_id}_{int(time.time())}"
        key = f"output:{prediction_id}"
        
        # Generate a random vector if not provided
        if "vector" in request.parameters:
            vector = request.parameters["vector"]
        else:
            vector = np.random.randn(384).tolist()  # Using 384 as standard vector size
        
        # Generate output using the model
        output_text = f"Generated output for query: {request.query}"
        confidence = 0.8  # Placeholder confidence score
        
        # Store in Redis
        result = {
            "text": output_text,
            "vector": vector,
            "confidence": confidence,
            "model_id": request.model_id,
            "query": request.query,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        redis_client.set(key, json.dumps(result))
        
        # Record metrics
        OUTPUT_GENERATED.labels(confidence_level="high" if confidence > 0.7 else "medium").inc()
        PROCESSING_LATENCY.observe(time.time() - start_time)
        
        return OutputResponse(
            prediction_id=prediction_id,
            results=[{"text": output_text}],
            metadata={
                "model_id": request.model_id,
                "confidence": confidence,
                "vector_size": len(vector)
            }
        )
    except Exception as e:
        logger.error(f"Output generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    os.environ["CUDA_VISIBLE_DEVICES"] = ""