import uvicorn
import os
import logging
from prometheus_client import start_http_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("masterchat")

def main():
    # Start Prometheus metrics server
    prometheus_port = int(os.getenv("PROMETHEUS_PORT", "8001"))
    start_http_server(prometheus_port)
    logger.info(f"Started Prometheus metrics server on port {prometheus_port}")

    # Start FastAPI server
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    logger.info(f"Starting MasterChat server on {host}:{port}")
    uvicorn.run(
        "lumina_core.masterchat.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main() 