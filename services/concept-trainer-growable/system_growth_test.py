import asyncio
import httpx
import docker
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List
import prometheus_client
from prometheus_client.parser import text_string_to_metric_families
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('growth_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
HOST = "localhost"
PORT = 8710
BASE_URL = f"http://{HOST}:{PORT}"
CHECK_INTERVAL = 300  # 5 minutes
MAX_RETRIES = 3

async def wait_for_service(client: httpx.AsyncClient, timeout: int = 60) -> bool:
    """Wait for the service to become available"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = await client.get(f"{BASE_URL}/health")
            if response.status_code == 200:
                logger.info("Service is available")
                return True
        except Exception as e:
            logger.warning(f"Service not yet available: {e}")
        await asyncio.sleep(2)
    return False

async def monitor_system_metrics(client: httpx.AsyncClient) -> Dict:
    """Monitor system-wide metrics from Prometheus"""
    try:
        response = await client.get(f"{BASE_URL}/metrics")
        if response.status_code != 200:
            logger.error(f"Failed to get metrics: {response.status_code}")
            return {}
            
        metrics_text = response.text
        metrics = {}
        for family in text_string_to_metric_families(metrics_text):
            for sample in family.samples:
                if any(key in sample.name for key in ['model_growth_events_total', 'layer_size', 'concept_drift', 'service_inputs_total']):
                    metrics[sample.name] = sample.value
        return metrics
    except Exception as e:
        logger.error(f"Error monitoring metrics: {e}")
        return {}

async def monitor_docker_stats() -> Dict:
    """Monitor Docker container stats"""
    try:
        docker_client = docker.from_env()
        stats = {}
        
        for container in docker_client.containers.list():
            if 'concept-trainer' in container.name:
                stats[container.name] = container.stats(stream=False)
        
        return stats
    except Exception as e:
        logger.error(f"Error monitoring Docker stats: {e}")
        return {}

async def test_system_growth(duration_hours: int = 24):
    """Test system's ability to grow through natural interactions"""
    logger.info(f"\n=== Starting System Growth Test for {duration_hours} hours ===")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Wait for service to be available
        if not await wait_for_service(client):
            logger.error("Service not available after timeout")
            return

        # Initial system state
        try:
            response = await client.get(f"{BASE_URL}/monitoring/network/stats")
            if response.status_code == 200:
                initial_stats = response.json()
                logger.info("\n=== Initial Network State ===")
                logger.info(f"Layer sizes: {initial_stats['layer_sizes']}")
                logger.info(f"Total parameters: {initial_stats['total_params']}")
                logger.info("===========================\n")
            else:
                logger.error(f"Failed to get initial stats: {response.status_code}")
                initial_stats = {"layer_sizes": [], "total_params": 0}
        except Exception as e:
            logger.error(f"Error getting initial stats: {e}")
            initial_stats = {"layer_sizes": [], "total_params": 0}
        
        # Monitor system for specified duration
        test_duration = timedelta(hours=duration_hours)
        start_time = datetime.now()
        last_check = start_time
        
        logger.info(f"\nMonitoring system for {test_duration.total_seconds()} seconds...")
        
        while datetime.now() - start_time < test_duration:
            try:
                # Get current network stats
                response = await client.get(f"{BASE_URL}/monitoring/network/stats")
                if response.status_code == 200:
                    current_stats = response.json()
                    
                    # Check for growth
                    if current_stats['growth_events'] > 0:
                        logger.info("\n🌱 GROWTH DETECTED! 🌱")
                        logger.info(f"Current layer sizes: {current_stats['layer_sizes']}")
                        logger.info(f"Growth ratios: {current_stats['growth_ratios']}")
                        logger.info(f"Total growth factor: {current_stats['total_growth_factor']}x")
                        logger.info(f"Total parameters: {current_stats['total_params']}")
                        
                        # Get detailed growth history
                        growth_response = await client.get(f"{BASE_URL}/monitoring/growth")
                        if growth_response.status_code == 200:
                            growth_data = growth_response.json()
                            logger.info("\nGrowth History:")
                            for event in growth_data['events'][-5:]:  # Show last 5 events
                                logger.info(f"Layer {event['layer_idx']}: {event['old_size']} -> {event['new_size']} (x{event['growth_factor']})")
                    
                    # Log periodic status
                    if (datetime.now() - last_check).total_seconds() >= 3600:  # Every hour
                        logger.info("\n=== Hourly Status Update ===")
                        logger.info(f"Time elapsed: {datetime.now() - start_time}")
                        logger.info(f"Current layer sizes: {current_stats['layer_sizes']}")
                        logger.info(f"Total parameters: {current_stats['total_params']}")
                        logger.info(f"Growth events: {current_stats['growth_events']}")
                        logger.info("==========================\n")
                        last_check = datetime.now()
                
            except Exception as e:
                logger.error(f"Error during monitoring cycle: {e}")
            
            # Wait before next check
            await asyncio.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    # Run the test for 24 hours with 5-minute check intervals
    asyncio.run(test_system_growth(duration_hours=24)) 