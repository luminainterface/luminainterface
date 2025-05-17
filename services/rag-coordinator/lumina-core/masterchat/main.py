from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import redis.asyncio as redis
import duckdb
import json
import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from sse_starlette.sse import EventSourceResponse
import os
import yaml
from jinja2 import Environment, FileSystemLoader
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from lumina_core.masterchat.llm_client import chat, get_token_count, get_cost_estimate
import time

# Import agents and intent routing
from lumina_core.masterchat.agents import get_agent, PLANS
from lumina_core.masterchat.agents.crawl_agent import CrawlAgent
from lumina_core.masterchat.agents.summarise_agent import SummariseAgent
from lumina_core.masterchat.agents.qa_agent import QAAgent
from lumina_core.masterchat.agents.orchestrator_agent import OrchestratorAgent
from lumina_core.masterchat.intent import route_intent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("masterchat")

# Environment variables with defaults
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
PLAN_INTERVAL_MIN = int(os.getenv("PLAN_INTERVAL_MIN", "10"))
MAX_CRAWL_NODES = int(os.getenv("MAX_CRAWL_NODES", "2000"))
LLM_BUDGET_USD = float(os.getenv("LLM_BUDGET_USD", "0.05"))
LLM_TEMP = float(os.getenv("LLM_TEMP", "0.3"))
GOAL_FILE = os.getenv("GOAL_FILE", "goal_lattice.yml")
PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", "8001"))

# Get the directory containing main.py
current_dir = os.path.dirname(os.path.abspath(__file__))

# Prometheus metrics
tasks_created = Counter('masterchat_tasks_total', 'Tasks by type', ['kind'])
plan_latency = Histogram('masterchat_plan_seconds', 'Planning cycle duration')
entropy_gauge = Gauge('graph_entropy', 'Current graph entropy')
tokens_used = Counter('masterchat_tokens_total', 'Tokens used', ['operation'])

app = FastAPI(
    title="MasterChat",
    root_path="/masterchat"  # Add root path to handle the /masterchat prefix
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis connection
redis_client = redis.from_url(REDIS_URL)

# DuckDB connection for metrics
db = duckdb.connect(":memory:")
db.execute("""
    CREATE TABLE IF NOT EXISTS events (
        timestamp TIMESTAMP,
        event_type STRING,
        data JSON
    )
""")

# Load goal lattice
try:
    goal_file_path = os.path.join(current_dir, GOAL_FILE)
    with open(goal_file_path) as f:
        goal_lattice = yaml.safe_load(f)
except FileNotFoundError:
    logger.error(f"Goal file not found: {GOAL_FILE}")
    goal_lattice = {"goals": [], "constraints": []}

# Initialize orchestrator agent
orchestrator = OrchestratorAgent()

# State
current_metrics = {
    "node_count": 0,
    "edge_count": 0,
    "delta_nodes": 0,
    "delta_edges": 0,
    "entropy": 0.0,
    "cluster_health": {}
}

# Planner log bus
planner_log_bus: List[str] = []

def log_planner(message: str, status: str = "info"):
    """Add message to planner log bus"""
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "message": message,
        "status": status
    }
    planner_log_bus.append(json.dumps(log_entry))
    # Keep only last 1000 messages
    if len(planner_log_bus) > 1000:
        planner_log_bus.pop(0)
    # Also send to Redis for event-mux
    asyncio.create_task(redis_client.xadd("planner_log", {"log": json.dumps(log_entry)}))

async def ingest():
    """Ingest events from Redis stream"""
    try:
        # Create consumer group if it doesn't exist
        try:
            await redis_client.xgroup_create("graph_stream", "mc", id="$", mkstream=True)
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise
            logger.info("Consumer group already exists")
            
        while True:
            try:
                # Read new events
                events = await redis_client.xreadgroup(
                    "mc", "mc", {"graph_stream": ">"}, count=1, block=1000
                )
                
                if events:
                    for stream, messages in events:
                        for msg_id, data in messages:
                            # Process event
                            event_type = data.get(b"type", b"").decode()
                            event_data = json.loads(data.get(b"data", b"{}"))
                            
                            # Log to DuckDB
                            db.execute(
                                "INSERT INTO events VALUES (?, ?, ?)",
                                (datetime.now(), event_type, json.dumps(event_data))
                            )
                            
                            # Update metrics
                            if event_type == "node.add":
                                current_metrics["node_count"] += 1
                                current_metrics["delta_nodes"] += 1
                            elif event_type == "edge.add":
                                current_metrics["edge_count"] += 1
                                current_metrics["delta_edges"] += 1
                            
                            # Acknowledge message
                            await redis_client.xack(stream, "mc", msg_id)
                            
            except Exception as e:
                logger.error(f"Error processing event: {e}")
                await asyncio.sleep(1)
                
    except Exception as e:
        logger.error(f"Event ingestion error: {e}")
        raise

# Task dispatch
async def dispatch_tasks(plan: dict):
    """Dispatch tasks to Redis stream"""
    tasks_stream = "tasks"
    try:
        await redis_client.xadd(tasks_stream, {"task": json.dumps(plan)}, maxlen=10000)
    except Exception as e:
        logger.error(f"Task dispatch error: {e}")

async def execute_plan(plan: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a plan step by step"""
    results = {}
    
    for step in plan:
        agent_name = step["agent"]
        agent = get_agent(agent_name)
        
        # Format input with context
        input_data = {}
        for key, value in step["input"].items():
            if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
                context_key = value[1:-1]
                input_data[key] = context.get(context_key)
            else:
                input_data[key] = value
        
        # Execute agent with retries
        log_planner(f"Executing {agent_name}: {step['description']}")
        for attempt in range(3):
            try:
                result = await agent.run(input_data)
                log_planner(f"{agent_name} completed successfully")
                break
            except Exception as e:
                if attempt == 2:  # Last attempt
                    log_planner(f"{agent_name} failed: {str(e)}", "error")
                    raise
                log_planner(f"{agent_name} attempt {attempt + 1} failed, retrying...", "warn")
                await asyncio.sleep(2 * attempt)
        
        # Store result in context
        context[f"{agent_name.lower()}_result"] = result
        results[agent_name] = result
        
        # Update metrics
        tasks_created.labels(kind=agent_name).inc()
    
    return results

class Task(BaseModel):
    type: str
    payload: Dict[str, Any]

class OverrideRequest(BaseModel):
    task_json: Dict[str, Any]

class ChatRequest(BaseModel):
    message: str

class TaskRequest(BaseModel):
    crawl: List[str]
    hops: int = 0
    max_nodes: int = 5

@app.post("/masterchat/chat")
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint with intent routing"""
    try:
        # Try intent routing first
        intent_result = await route_intent(request.message)
        
        if intent_result:
            # Use predefined plan based on intent
            plan = PLANS[intent_result["mode"]]
            context = intent_result["payload"]
        else:
            # Fall back to LLM planner
            log_planner("No intent detected, using LLM planner")
            # TODO: Implement LLM planner fallback
            raise HTTPException(status_code=501, detail="LLM planner not implemented yet")
        
        # Execute plan
        results = await execute_plan(plan, context)
        
        # Return final answer
        return results["QAAgent"]
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tasks")
async def submit_task(request: TaskRequest):
    """Submit a new task to the queue"""
    try:
        # Create task payload
        task = {
            "crawl": request.crawl,
            "hops": request.hops,
            "max_nodes": request.max_nodes,
            "ui": {"severity": "info", "msg": "Task submitted"},
            "note": "manual task"
        }
        
        # Add to Redis stream
        await redis_client.xadd("tasks", {"task": json.dumps(task)}, maxlen=10000)
        
        # Log to DuckDB
        db.execute("INSERT INTO planner_logs VALUES (?,?)", 
                  (time.time(), json.dumps(task)))
        
        return {"status": "ok", "task": task}
    except Exception as e:
        logger.error(f"Task submission error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check() -> Dict[str, bool]:
    try:
        redis_client.ping()
        return {"planner_alive": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/planner/logs")
async def watch_logs(request: Request):
    """SSE endpoint for planner logs"""
    async def event_generator():
        idx = 0
        while True:
            if await request.is_disconnected():
                break
            while idx < len(planner_log_bus):
                yield planner_log_bus[idx]
                idx += 1
            await asyncio.sleep(0.5)
    
    return EventSourceResponse(event_generator())

async def run_orchestrator_cycle():
    """Run orchestrator cycle to check and update system state"""
    try:
        with plan_latency.time():
            # Get current state from Redis
            state = await redis_client.get("system_state")
            state = json.loads(state) if state else {}
            
            # Run orchestrator
            result = await orchestrator.run({"state": state})
            
            # Update metrics
            if result.get("status") == "success":
                entropy_gauge.set(result.get("entropy", 0))
                current_metrics.update(result.get("metrics", {}))
                
                # Log changes
                if result.get("changes"):
                    log_planner(f"Changes detected: {result['changes']}")
                    
                # Store updated state
                await redis_client.set("system_state", json.dumps(result.get("state", {})))
            else:
                log_planner(f"Orchestrator cycle failed: {result.get('error')}", "error")
    except Exception as e:
        logger.error(f"Error in orchestrator cycle: {e}")
        log_planner(f"Orchestrator error: {str(e)}", "error")

@app.on_event("startup")
async def startup_event():
    """Start background tasks on startup"""
    # Start Prometheus metrics server
    start_http_server(PROMETHEUS_PORT)
    logger.info(f"Started Prometheus metrics server on port {PROMETHEUS_PORT}")
    
    # Start event ingestion loop
    asyncio.create_task(ingest())
    
    # Start orchestrator cycle
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        run_orchestrator_cycle,
        trigger=IntervalTrigger(minutes=PLAN_INTERVAL_MIN),
        id='orchestrator_cycle',
        name='Run orchestrator cycle',
        replace_existing=True
    )
    scheduler.start()
    
    logger.info(f"Starting MasterChat server on 0.0.0.0:8000") 