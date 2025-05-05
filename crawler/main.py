import asyncio
import uvicorn
from metrics_server import app, pages_total, errors_total, duration
from contextlib import asynccontextmanager

async def crawl_loop():
    while True:
        try:
            # Your existing crawl logic here
            with duration.time():
                # Your page fetch logic
                pages_total.inc()
        except Exception as e:
            errors_total.inc()
            print(f"Error in crawl loop: {e}")
        await asyncio.sleep(1)

@asynccontextmanager
async def lifespan(app):
    # Start the crawler loop in the background
    asyncio.create_task(crawl_loop())
    yield

app.router.lifespan_context = lifespan

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8400) 