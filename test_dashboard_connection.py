#!/usr/bin/env python3
"""
Test Dashboard Connection

Simple script to test socket connection for LUMINA V7 dashboard.
"""

import socket
import json
import threading
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TestConnection")

def handle_client(client_socket, address):
    """Handle a client connection"""
    logger.info(f"Client connected from {address}")
    
    try:
        while True:
            # Receive data
            data = client_socket.recv(1024)
            if not data:
                break
                
            # Try to parse as JSON
            try:
                request = json.loads(data.decode('utf-8'))
                logger.info(f"Received request: {request}")
                
                # If it's a metrics request, send back mock metrics
                if request.get("type") == "metrics_request":
                    response = {
                        "metrics": {
                            "consciousness_level": 0.7,
                            "mistral_activity": 0.8,
                            "learning_rate": 0.5,
                            "system_usage": 0.4
                        },
                        "descriptions": {
                            "consciousness_level": "Mock consciousness level",
                            "mistral_activity": "Mock Mistral activity",
                            "learning_rate": "Mock learning rate",
                            "system_usage": "Mock system usage"
                        },
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
                    }
                    
                    # Send response
                    client_socket.sendall(json.dumps(response).encode('utf-8'))
                    logger.info("Sent mock metrics response")
            except json.JSONDecodeError:
                logger.warning(f"Received non-JSON data: {data}")
    except Exception as e:
        logger.error(f"Error handling client: {e}")
    finally:
        client_socket.close()
        logger.info(f"Connection from {address} closed")

def main():
    """Main entry point"""
    # Create server socket
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # Bind to port
    server_address = ('localhost', 5678)
    logger.info(f"Starting server on {server_address}")
    server.bind(server_address)
    
    # Listen for connections
    server.listen(5)
    logger.info("Server is listening for connections")
    
    try:
        while True:
            # Accept connection
            client_socket, address = server.accept()
            
            # Handle in a separate thread
            client_thread = threading.Thread(
                target=handle_client,
                args=(client_socket, address)
            )
            client_thread.daemon = True
            client_thread.start()
    except KeyboardInterrupt:
        logger.info("Server shutting down")
    finally:
        server.close()

if __name__ == "__main__":
    main() 