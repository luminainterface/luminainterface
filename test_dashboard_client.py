#!/usr/bin/env python3
"""
Test Dashboard Client

Simple script to test connecting to the LUMINA V7 dashboard server.
"""

import socket
import json
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TestClient")

def main():
    """Main entry point"""
    # Create client socket
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Connect to server
    server_address = ('localhost', 5678)
    logger.info(f"Connecting to server on {server_address}")
    
    try:
        client.connect(server_address)
        logger.info("Connected to server")
        
        # Create metrics request
        request = {
            "type": "metrics_request",
            "metrics": ["consciousness_level", "mistral_activity", "learning_rate", "system_usage"]
        }
        
        # Send request
        logger.info(f"Sending request: {request}")
        client.sendall(json.dumps(request).encode('utf-8'))
        
        # Receive response
        response = client.recv(4096)
        if response:
            try:
                response_data = json.loads(response.decode('utf-8'))
                logger.info(f"Received response: {response_data}")
                
                # Display metrics
                if "metrics" in response_data:
                    metrics = response_data["metrics"]
                    logger.info("\nMetrics:")
                    for metric, value in metrics.items():
                        description = response_data.get("descriptions", {}).get(metric, "")
                        logger.info(f"  {metric}: {value} - {description}")
            except json.JSONDecodeError:
                logger.error(f"Received invalid JSON response: {response}")
        else:
            logger.warning("No response received from server")
    except Exception as e:
        logger.error(f"Error connecting to server: {e}")
    finally:
        client.close()
        logger.info("Connection closed")

if __name__ == "__main__":
    main() 