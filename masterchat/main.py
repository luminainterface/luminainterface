import os
import json
import asyncio
import time
from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException
from sse_starlette.sse import EventSourceResponse
import redis.asyncio as redis
from redis.exceptions import ResponseError
import duckdb
import apscheduler.schedulers.asyncio
from mistralai.async_client import MistralAsyncClient
from prometheus_client import generate_latest, CollectorRegistry, Counter, Histogram
from lumina_core.common.cors import add_cors

app = FastAPI(title="MasterChat")
add_cors(app)

# Initialize metrics
reg = CollectorRegistry()
request_total = Counter("masterchat_requests_total", "REST requests", ["path"], registry=reg)
latency = Histogram("masterchat_request_seconds", "Latency", ["path"], registry=reg)
plan_seconds = Histogram("masterchat_plan_seconds", "Planner cycle latency", registry=reg)

# Initialize Redis client
redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379"))

db = duckdb.connect("/data/state.db")
db.execute("CREATE TABLE IF NOT EXISTS events(ts DOUBLE, type VARCHAR, body VARCHAR)")
db.execute("CREATE TABLE IF NOT EXISTS planner_logs(ts DOUBLE, msg VARCHAR)")

CLIENT = MistralAsyncClient(api_key=os.getenv("MISTRAL_API_KEY", ""))

# ---------- Ingest ---------- #
async def ingest():
    try:
        await redis_client.xgroup_create("graph_stream", "mc", id="$", mkstream=True)
    except ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise
        print("Consumer group already exists")
        
    while True:
        try:
            msgs = await redis_client.xreadgroup("mc", "mc-1", {"graph_stream": ">"}, block=5000, count=256)
            for stream, pairs in msgs:
                for msg_id, data in pairs:
                    evt = json.loads(data[b'event'])
                    db.execute("INSERT INTO events VALUES (?,?,?)",
                            (evt["ts"], evt["type"], json.dumps(evt)))
                    await redis_client.xack(stream, "mc", msg_id)
            await asyncio.sleep(0.2)
        except Exception as e:
            print(f"Error in ingest loop: {e}")
            await asyncio.sleep(1)  # Wait before retrying

# ---------- Planner ---------- #
async def plan_cycle():
    t0 = time.time()
    # quick metric snapshot
    d_nodes = db.execute(
        "SELECT COUNT(*) FROM events WHERE type='node.add' AND ts> ?",
        [time.time() - 600]).fetchone()[0]
    entropy = 0.5  # TODO: compute real value
    try:
        with open("prompts/plan_cycle.j2") as f:
            prompt = f.read().format(
                now=time.ctime(), d_nodes=d_nodes, entropy=entropy)
    except FileNotFoundError:
        print("Warning: plan_cycle.j2 not found, using default prompt")
        prompt = f"""Current time: {time.ctime()}
New nodes in last 10m: {d_nodes}
Graph entropy: {entropy}
What should we do next?"""

    try:
        if os.getenv("MISTRAL_API_KEY"):
            rsp = await CLIENT.chat(
                model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            plan = json.loads(rsp.choices[0].message.content)
        else:
            plan = {"crawl":["Mandelbrot set"],"prune":None,"focus":None,"ui":{"severity":"info","msg":"LLM off—rule mode"},"note":"fallback"}
    except Exception as e:
        plan = {"ui":{"severity":"error","msg":f"Planner error {e}"}}

    await redis_client.xadd("tasks", {"task": json.dumps(plan)}, maxlen=10000)
    db.execute("INSERT INTO planner_logs VALUES (?,?)", (time.time(), json.dumps(plan)))
    print("🧠 planned in", round(time.time()-t0,2), "s")

# ---------- SSE ---------- #
@app.get("/planner/logs")
async def planner_logs(req: Request):
    async def gen():
        cur = 0
        while True:
            if await req.is_disconnected(): break
            rows = db.execute("SELECT ts,msg FROM planner_logs WHERE ts> ? ORDER BY ts", [cur]).fetchall()
            for ts,msg in rows:
                cur = ts
                yield {"event":"log","data":msg}
            await asyncio.sleep(1)
    return EventSourceResponse(gen())

@app.get("/metrics")
def metrics():
    return generate_latest(reg), 200, {"Content-Type": "text/plain"}

@app.middleware("http")
async def metrics_middleware(request, call_next):
    path = request.url.path.split("?")[0]
    with latency.labels(path=path).time():
        response = await call_next(request)
    request_total.labels(path=path).inc()
    return response

@app.get("/health")
async def health_check() -> Dict[str, bool]:
    try:
        await redis_client.ping()
        return {"planner_alive": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- bootstrap ---------- #
@app.on_event("startup")
async def boot():
    loop = asyncio.get_event_loop()
    loop.create_task(ingest())
    sched = apscheduler.schedulers.asyncio.AsyncIOScheduler()
    cron = os.getenv("PLAN_CRON", "*/10 * * * *")
    sched.add_job(plan_cycle, trigger="cron", **dict(zip(
        ["minute","hour","day","month","day_of_week"], cron.split())))
    sched.start() 