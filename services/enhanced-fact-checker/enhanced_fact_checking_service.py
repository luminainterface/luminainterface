#!/usr/bin/env python3
"""
ENHANCED FACT-CHECKING SERVICE
=============================

Web service version of the Enhanced Fact-Checking Layer for Docker deployment.
Provides REST API endpoints for fact-checking functionality.
"""

import asyncio
import aiohttp
from aiohttp import web, web_request
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Import the enhanced fact checker (from same directory)
from enhanced_fact_checking_layer import EnhancedFactChecker

class EnhancedFactCheckingService:
    """Web service wrapper for Enhanced Fact Checker"""
    
    def __init__(self):
        self.fact_checker = None
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for the service"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger("FactCheckService")
    
    async def initialize(self):
        """Initialize the fact checker"""
        self.logger.info("Initializing Enhanced Fact Checker Service...")
        self.fact_checker = EnhancedFactChecker()
        await self.fact_checker.__aenter__()
        self.logger.info("✅ Enhanced Fact Checker Service initialized successfully")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.fact_checker:
            await self.fact_checker.__aexit__(None, None, None)
            self.logger.info("✅ Enhanced Fact Checker Service cleaned up")
    
    async def health_check(self, request: web_request.Request) -> web.Response:
        """Health check endpoint"""
        try:
            return web.json_response({
                "status": "healthy",
                "service": "enhanced-fact-checker",
                "version": "1.0.0",
                "port": 8885,
                "timestamp": datetime.now().isoformat(),
                "fact_checker_ready": self.fact_checker is not None
            })
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return web.json_response({
                "status": "unhealthy", 
                "error": str(e)
            }, status=500)
    
    async def fact_check_endpoint(self, request: web_request.Request) -> web.Response:
        """Main fact-checking endpoint"""
        try:
            # Parse request
            request_data = await request.json()
            text = request_data.get('text', '')
            
            if not text:
                return web.json_response({
                    "error": "Missing 'text' parameter"
                }, status=400)
            
            self.logger.info(f"Processing fact-check request for text: {text[:100]}...")
            
            start_time = time.time()
            
            # Perform fact checking
            enhanced_response, fact_results = await self.fact_checker.comprehensive_fact_check(text)
            
            processing_time = time.time() - start_time
            
            # Prepare response
            response_data = {
                "original_text": text,
                "enhanced_text": enhanced_response,
                "processing_time": processing_time,
                "claims_processed": len(fact_results),
                "fact_check_results": [
                    {
                        "claim": fr.claim,
                        "is_accurate": fr.is_accurate,
                        "confidence_score": fr.confidence_score,
                        "corrections": fr.corrections,
                        "sources": fr.sources,
                        "verification_method": fr.verification_method
                    } for fr in fact_results
                ],
                "accuracy_summary": {
                    "total_claims": len(fact_results),
                    "accurate_claims": sum(1 for fr in fact_results if fr.is_accurate),
                    "accuracy_rate": sum(1 for fr in fact_results if fr.is_accurate) / max(len(fact_results), 1),
                    "corrections_applied": sum(1 for fr in fact_results if fr.corrections)
                },
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Fact-check completed in {processing_time:.2f}s with {len(fact_results)} claims")
            
            return web.json_response(response_data)
            
        except Exception as e:
            self.logger.error(f"Fact-checking failed: {e}")
            return web.json_response({
                "error": f"Fact-checking failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }, status=500)
    
    async def batch_fact_check_endpoint(self, request: web_request.Request) -> web.Response:
        """Batch fact-checking endpoint for multiple texts"""
        try:
            request_data = await request.json()
            texts = request_data.get('texts', [])
            
            if not texts or not isinstance(texts, list):
                return web.json_response({
                    "error": "Missing 'texts' parameter or not a list"
                }, status=400)
            
            self.logger.info(f"Processing batch fact-check request for {len(texts)} texts")
            
            start_time = time.time()
            batch_results = []
            
            for i, text in enumerate(texts):
                try:
                    enhanced_response, fact_results = await self.fact_checker.comprehensive_fact_check(text)
                    
                    batch_results.append({
                        "index": i,
                        "original_text": text,
                        "enhanced_text": enhanced_response,
                        "claims_processed": len(fact_results),
                        "accuracy_rate": sum(1 for fr in fact_results if fr.is_accurate) / max(len(fact_results), 1),
                        "corrections_applied": sum(1 for fr in fact_results if fr.corrections),
                        "fact_check_results": [
                            {
                                "claim": fr.claim,
                                "is_accurate": fr.is_accurate,
                                "confidence_score": fr.confidence_score,
                                "corrections": fr.corrections,
                                "sources": fr.sources
                            } for fr in fact_results
                        ]
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error processing text {i}: {e}")
                    batch_results.append({
                        "index": i,
                        "original_text": text,
                        "error": str(e)
                    })
            
            total_processing_time = time.time() - start_time
            
            response_data = {
                "batch_results": batch_results,
                "batch_summary": {
                    "total_texts": len(texts),
                    "successful_processes": len([r for r in batch_results if "error" not in r]),
                    "failed_processes": len([r for r in batch_results if "error" in r]),
                    "total_processing_time": total_processing_time,
                    "average_processing_time": total_processing_time / len(texts)
                },
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Batch fact-check completed in {total_processing_time:.2f}s")
            
            return web.json_response(response_data)
            
        except Exception as e:
            self.logger.error(f"Batch fact-checking failed: {e}")
            return web.json_response({
                "error": f"Batch fact-checking failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }, status=500)
    
    async def service_info_endpoint(self, request: web_request.Request) -> web.Response:
        """Service information endpoint"""
        return web.json_response({
            "service": "enhanced-fact-checker",
            "version": "1.0.0",
            "description": "Enhanced fact-checking layer with LoRA correction and NLP-RAG enhancement",
            "endpoints": {
                "/health": "Health check",
                "/fact-check": "Single text fact-checking",
                "/batch-fact-check": "Batch text fact-checking",
                "/info": "Service information"
            },
            "features": [
                "Scientific claim extraction",
                "Database fact verification", 
                "LoRA correction enhancement",
                "RAG integration",
                "Multi-domain support"
            ],
            "supported_domains": [
                "Geography & Politics",
                "Scientific accuracy",
                "Mathematical concepts",
                "Historical events",
                "Technology concepts",
                "Medical & health",
                "Astronomy & space",
                "Arts & culture"
            ],
            "timestamp": datetime.now().isoformat()
        })

    def create_app(self) -> web.Application:
        """Create the web application"""
        app = web.Application()
        
        # Add routes
        app.router.add_get('/health', self.health_check)
        app.router.add_get('/info', self.service_info_endpoint)
        app.router.add_post('/fact-check', self.fact_check_endpoint)
        app.router.add_post('/batch-fact-check', self.batch_fact_check_endpoint)
        
        # Add CORS headers
        async def add_cors_headers(request, handler):
            response = await handler(request)
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            return response
        
        app.middlewares.append(add_cors_headers)
        
        return app

async def main():
    """Main service entry point"""
    service = EnhancedFactCheckingService()
    
    try:
        # Initialize service
        await service.initialize()
        
        # Create web app
        app = service.create_app()
        
        # Setup cleanup on shutdown
        async def cleanup_on_shutdown(app):
            await service.cleanup()
        
        app.on_cleanup.append(cleanup_on_shutdown)
        
        # Start server
        runner = web.AppRunner(app)
        await runner.setup()
        
        site = web.TCPSite(runner, '0.0.0.0', 8885)
        await site.start()
        
        service.logger.info("🌟 Enhanced Fact-Checking Service started on http://0.0.0.0:8885")
        service.logger.info("📋 Available endpoints:")
        service.logger.info("   GET  /health - Health check")
        service.logger.info("   GET  /info - Service information")
        service.logger.info("   POST /fact-check - Single text fact-checking")
        service.logger.info("   POST /batch-fact-check - Batch text fact-checking")
        
        # Keep running
        while True:
            await asyncio.sleep(3600)  # Sleep for 1 hour
            
    except KeyboardInterrupt:
        service.logger.info("🛑 Service stopped by user")
    except Exception as e:
        service.logger.error(f"❌ Service error: {e}")
    finally:
        await service.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 