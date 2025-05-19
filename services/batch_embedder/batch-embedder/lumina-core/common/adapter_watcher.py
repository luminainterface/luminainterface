import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Callable
import httpx
import shutil
from prometheus_client import Gauge

from lumina_core.common.bus import BusClient, StreamMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("adapter-watcher")

# Initialize metrics
ADAPTER_VERSION = Gauge(
    "adapter_version",
    "Current adapter ID as numeric hash",
    ["service"]
)

class AdapterUpdate(StreamMessage):
    adapter_id: str
    ts: float

async def watch_and_reload(
    load_fn: Callable[[str], None],
    adapter_dir: str | Path,
    service_name: str = None
) -> None:
    """
    Watch for adapter updates and reload them into the model.
    
    Args:
        load_fn: Function that loads an adapter into the running model
        adapter_dir: Directory to store adapter files
        service_name: Optional service name for metrics (defaults to HOSTNAME)
    """
    bus = BusClient(redis_url=os.getenv("REDIS_URL", "redis://redis:6379"))
    adapter_dir = Path(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    
    service_name = service_name or os.getenv("HOSTNAME", "inference")
    
    async def download_adapter(adapter_id: str) -> Path:
        """Download adapter file from trainer service"""
        path = adapter_dir / f"{adapter_id}.bin"
        if path.exists():
            logger.info(f"Adapter {adapter_id} already exists at {path}")
            return path
            
        logger.info(f"Downloading adapter {adapter_id}...")
        url = f"http://concept-trainer-growable:8710/adapters/{adapter_id}"
        
        try:
            async with httpx.AsyncClient(timeout=180) as client:
                async with client.stream("GET", url) as response:
                    response.raise_for_status()
                    with open(path, "wb") as f:
                        async for chunk in response.aiter_bytes():
                            f.write(chunk)
            logger.info(f"Downloaded adapter {adapter_id} to {path}")
            return path
        except Exception as e:
            logger.error(f"Failed to download adapter {adapter_id}: {e}")
            raise
    
    async def handle_update(msg: dict):
        """Handle adapter update message"""
        try:
            update = AdapterUpdate(**msg)
            adapter_id = update.adapter_id
            
            # Download if needed
            path = await download_adapter(adapter_id)
            
            # Load into model
            logger.info(f"Loading adapter {adapter_id} into model...")
            load_fn(str(path))
            
            # Update metric
            version_hash = int(hash(adapter_id) & 0xffffffff)
            ADAPTER_VERSION.labels(service=service_name).set(version_hash)
            logger.info(f"Adapter {adapter_id} loaded (version hash: {version_hash})")
            
        except Exception as e:
            logger.error(f"Error handling adapter update: {e}")
            raise
    
    try:
        # Connect to Redis
        await bus.connect()
        
        # Start consumer
        logger.info(f"Starting adapter watcher for {service_name}...")
        await bus.consume(
            stream="model.adapter.updated",
            group="inference",
            consumer=service_name,
            handler=handle_update,
            block_ms=1000,
            count=1
        )
    except Exception as e:
        logger.error(f"Adapter watcher error: {e}")
        raise
    finally:
        await bus.close() 