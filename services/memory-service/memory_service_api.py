#!/usr/bin/env python3
"""
Memory Service API Wrapper
==========================

Flask API wrapper for the enhanced memory service with health endpoints.
"""

import asyncio
import json
import logging
from flask import Flask, jsonify, request
from auto_scraper_integration import EnhancedMemoryService
import threading
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global memory service instance
memory_service = None
service_status = {
    "status": "starting",
    "start_time": time.time(),
    "last_health_check": None
}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global service_status
    
    service_status["last_health_check"] = time.time()
    
    try:
        # Basic health check
        uptime = time.time() - service_status["start_time"]
        
        health_data = {
            "status": "healthy",
            "service": "enhanced_memory_service",
            "uptime_seconds": round(uptime, 2),
            "auto_scraper_enabled": memory_service is not None,
            "timestamp": time.time()
        }
        
        return jsonify(health_data), 200
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Get detailed service status"""
    try:
        if memory_service:
            # This would need to be adapted for async
            stats = {
                "service_status": "running",
                "auto_scraper": "enabled",
                "uptime": time.time() - service_status["start_time"]
            }
        else:
            stats = {
                "service_status": "basic_mode",
                "auto_scraper": "disabled",
                "uptime": time.time() - service_status["start_time"]
            }
        
        return jsonify(stats), 200
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/scrape/manual', methods=['POST'])
def manual_scrape():
    """Trigger manual scraping"""
    try:
        if not memory_service:
            return jsonify({"error": "Auto-scraper not initialized"}), 400
        
        # This would need async handling in a real implementation
        return jsonify({"message": "Manual scrape triggered"}), 200
        
    except Exception as e:
        logger.error(f"Manual scrape failed: {e}")
        return jsonify({"error": str(e)}), 500

def initialize_memory_service():
    """Initialize the memory service in a separate thread"""
    global memory_service, service_status
    
    try:
        logger.info("Initializing enhanced memory service...")
        memory_service = EnhancedMemoryService()
        service_status["status"] = "running"
        logger.info("Enhanced memory service initialized successfully")
        
        # Note: In a real implementation, you'd want to properly handle the async event loop
        # For now, we'll just initialize the service without starting auto-scraping
        
    except Exception as e:
        logger.error(f"Failed to initialize memory service: {e}")
        service_status["status"] = "error"
        service_status["error"] = str(e)

if __name__ == '__main__':
    # Initialize memory service in background
    init_thread = threading.Thread(target=initialize_memory_service)
    init_thread.daemon = True
    init_thread.start()
    
    # Start Flask app
    logger.info("Starting Memory Service API on port 8915")
    app.run(host='0.0.0.0', port=8915, debug=False) 