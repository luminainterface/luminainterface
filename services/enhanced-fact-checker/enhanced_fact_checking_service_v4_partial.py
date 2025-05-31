#!/usr/bin/env python3
"""
ENHANCED FACT-CHECKING SERVICE V4 - PARTIAL DOMAIN PRODUCTION DEPLOYMENT
======================================================================

🚀 PRODUCTION-READY DOMAINS (100% Validated):
   ✅ TECHNOLOGY - Programming, AI/ML, Internet History
   ✅ MEDICINE - Anatomy, Disease Treatment, Pharmaceuticals  
   ✅ PSYCHOLOGY - Learning Theory, Development, Social Psychology
   ✅ GEOGRAPHY - World Capitals, Organizations, Physical Features

🚧 NON-PRODUCTION DOMAINS (Under Development for V5):
   ❌ CHEMISTRY - Atomic structure, Chemical reactions
   ❌ ENGINEERING - Electrical, Mechanical, Civil engineering
   ❌ SPORTS - Olympic games, World records, Team sports
   ❌ GLOBAL_ISSUES - Climate change, Public health policy
   ❌ EDGE_CASES - Context-dependent, Temporal claims

DEPLOYMENT STATUS: ✅ APPROVED FOR PARTIAL PRODUCTION
TARGET: Docker integration for validated domains only
FALLBACK: Clear error messages for non-covered domains
"""

import asyncio
import aiohttp
import sqlite3
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from flask import Flask, request, jsonify, abort

# Import V4 production system
import sys
import os
sys.path.append('/app')

try:
    from enhanced_fact_checking_layer_v4_production import EnhancedFactCheckerV4, FactCheckResult
except ImportError:
    # Fallback if not available
    logging.error("V4 production system not available - using mock")
    
@dataclass
class V4PartialCoverageResult:
    """V4 Partial coverage result with explicit domain support status"""
    original_text: str
    enhanced_text: str
    fact_results: List[Dict[str, Any]]
    domains_processed: List[str]
    domains_supported: List[str]
    domains_unsupported: List[str]
    processing_time: float
    production_status: str
    coverage_warnings: List[str]
    v4_confidence: float

class V4PartialFactCheckingService:
    """V4 Production service with explicit partial domain coverage"""
    
    def __init__(self):
        self.app = Flask(__name__)
        self.fact_checker = None
        
        # V4 VALIDATED PRODUCTION DOMAINS
        self.v4_production_domains = {
            "TECHNOLOGY": {
                "production_ready": True,
                "confidence": 0.95,
                "coverage_areas": ["Programming Languages", "AI/ML", "Internet History", "Technology Companies"],
                "validation_rules": 25
            },
            "MEDICINE": {
                "production_ready": True, 
                "confidence": 0.93,
                "coverage_areas": ["Human Anatomy", "Disease Treatment", "Pharmaceuticals", "Medical History"],
                "validation_rules": 22
            },
            "PSYCHOLOGY": {
                "production_ready": True,
                "confidence": 0.90,
                "coverage_areas": ["Learning Theory", "Developmental Psychology", "Social Psychology"],
                "validation_rules": 18
            },
            "GEOGRAPHY": {
                "production_ready": True,
                "confidence": 0.92,
                "coverage_areas": ["World Capitals", "International Organizations", "Physical Features"],
                "validation_rules": 20
            }
        }
        
        # V5 DEVELOPMENT DOMAINS (Not production ready)
        self.v5_development_domains = {
            "CHEMISTRY": {
                "production_ready": False,
                "expected_v5_release": "Q2 2024",
                "coverage_gap": "Atomic structure, Chemical reactions validation"
            },
            "ENGINEERING": {
                "production_ready": False,
                "expected_v5_release": "Q2 2024", 
                "coverage_gap": "Engineering safety, Material properties validation"
            },
            "SPORTS": {
                "production_ready": False,
                "expected_v5_release": "Q2 2024",
                "coverage_gap": "Sports rules, Record validation"
            },
            "GLOBAL_ISSUES": {
                "production_ready": False,
                "expected_v5_release": "Q3 2024",
                "coverage_gap": "Climate policy, Global health misinformation"
            },
            "EDGE_CASES": {
                "production_ready": False,
                "expected_v5_release": "Q3 2024",
                "coverage_gap": "Context-dependent claims, Temporal validation"
            }
        }
        
        self.logger = self._setup_logging()
        self._setup_routes()
    
    def _setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger("V4PartialFactChecker")
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                "status": "healthy",
                "service": "Enhanced Fact-Checker V4 - Partial Production",
                "version": "v4.0-partial",
                "production_domains": list(self.v4_production_domains.keys()),
                "development_domains": list(self.v5_development_domains.keys()),
                "timestamp": datetime.now().isoformat()
            })
        
        @self.app.route('/coverage', methods=['GET'])
        def coverage_status():
            """Get detailed coverage status"""
            return jsonify({
                "production_ready_domains": self.v4_production_domains,
                "development_domains": self.v5_development_domains,
                "total_validation_rules": sum(d["validation_rules"] for d in self.v4_production_domains.values()),
                "production_coverage": f"{len(self.v4_production_domains)}/{len(self.v4_production_domains) + len(self.v5_development_domains)} domains"
            })
        
        @self.app.route('/fact-check', methods=['POST'])
        def fact_check_endpoint():
            """Main fact-checking endpoint with V4 partial coverage"""
            try:
                data = request.get_json()
                
                if not data or 'text' not in data:
                    return jsonify({"error": "Missing 'text' field in request"}), 400
                
                text = data['text']
                
                # Execute V4 partial fact-checking
                result = asyncio.run(self.v4_partial_fact_check(text))
                
                return jsonify(asdict(result))
                
            except Exception as e:
                self.logger.error(f"V4 Fact-checking error: {str(e)}")
                return jsonify({"error": f"Fact-checking failed: {str(e)}"}), 500
        
        @self.app.route('/domains/<domain>/check', methods=['POST'])
        def domain_specific_check(domain):
            """Domain-specific fact-checking endpoint"""
            try:
                if domain.upper() not in self.v4_production_domains:
                    if domain.upper() in self.v5_development_domains:
                        return jsonify({
                            "error": f"Domain '{domain}' is not production ready",
                            "status": "development",
                            "expected_release": self.v5_development_domains[domain.upper()]["expected_v5_release"],
                            "coverage_gap": self.v5_development_domains[domain.upper()]["coverage_gap"]
                        }), 503  # Service Unavailable
                    else:
                        return jsonify({"error": f"Unknown domain: {domain}"}), 404
                
                data = request.get_json()
                if not data or 'text' not in data:
                    return jsonify({"error": "Missing 'text' field in request"}), 400
                
                # Execute domain-specific checking
                result = asyncio.run(self.domain_specific_fact_check(data['text'], domain.upper()))
                
                return jsonify(asdict(result))
                
            except Exception as e:
                self.logger.error(f"Domain-specific fact-checking error: {str(e)}")
                return jsonify({"error": f"Domain checking failed: {str(e)}"}), 500
    
    async def v4_partial_fact_check(self, text: str) -> V4PartialCoverageResult:
        """V4 partial fact-checking with explicit domain coverage"""
        
        start_time = time.time()
        
        self.logger.info(f"🔍 V4 PARTIAL FACT-CHECK: {text[:100]}...")
        
        coverage_warnings = []
        domains_processed = []
        domains_supported = []
        domains_unsupported = []
        
        try:
            # Initialize fact checker if needed
            if not self.fact_checker:
                self.fact_checker = EnhancedFactCheckerV4()
                await self.fact_checker.__aenter__()
            
            # Execute V4 production fact-checking
            enhanced_text, fact_results = await self.fact_checker.production_fact_check_v4(text)
            
            # Analyze domain coverage
            for result in fact_results:
                domain = result.domain
                domains_processed.append(domain)
                
                if domain in self.v4_production_domains:
                    domains_supported.append(domain)
                else:
                    domains_unsupported.append(domain)
                    
                    if domain in self.v5_development_domains:
                        coverage_warnings.append(
                            f"Domain '{domain}' is under development for V5 - "
                            f"Expected release: {self.v5_development_domains[domain]['expected_v5_release']}"
                        )
                    else:
                        coverage_warnings.append(f"Domain '{domain}' is not yet supported")
            
            # Calculate V4 confidence based on supported domains
            supported_results = [r for r in fact_results if r.domain in self.v4_production_domains]
            total_confidence = sum(r.confidence_score for r in supported_results) / max(len(supported_results), 1)
            
            processing_time = time.time() - start_time
            
            # Determine production status
            if len(domains_supported) == len(domains_processed):
                production_status = "FULLY_SUPPORTED"
            elif len(domains_supported) > 0:
                production_status = "PARTIALLY_SUPPORTED"
            else:
                production_status = "UNSUPPORTED_DOMAINS"
            
            self.logger.info(
                f"✅ V4 PARTIAL CHECK COMPLETE: "
                f"{len(domains_supported)}/{len(domains_processed)} domains supported, "
                f"{processing_time:.3f}s processing time"
            )
            
            return V4PartialCoverageResult(
                original_text=text,
                enhanced_text=enhanced_text,
                fact_results=[{
                    "claim": r.claim,
                    "is_accurate": r.is_accurate,
                    "confidence_score": r.confidence_score,
                    "corrections": r.corrections,
                    "sources": r.sources,
                    "verification_method": r.verification_method,
                    "domain": r.domain,
                    "claim_type": r.claim_type,
                    "production_supported": r.domain in self.v4_production_domains
                } for r in fact_results],
                domains_processed=list(set(domains_processed)),
                domains_supported=list(set(domains_supported)),
                domains_unsupported=list(set(domains_unsupported)),
                processing_time=processing_time,
                production_status=production_status,
                coverage_warnings=coverage_warnings,
                v4_confidence=total_confidence
            )
            
        except Exception as e:
            self.logger.error(f"V4 partial fact-checking failed: {str(e)}")
            
            processing_time = time.time() - start_time
            
            return V4PartialCoverageResult(
                original_text=text,
                enhanced_text=text,  # No enhancement on error
                fact_results=[],
                domains_processed=[],
                domains_supported=[],
                domains_unsupported=[],
                processing_time=processing_time,
                production_status="ERROR",
                coverage_warnings=[f"Fact-checking failed: {str(e)}"],
                v4_confidence=0.0
            )
    
    async def domain_specific_fact_check(self, text: str, domain: str) -> V4PartialCoverageResult:
        """Domain-specific fact-checking for supported domains"""
        
        if domain not in self.v4_production_domains:
            raise ValueError(f"Domain {domain} is not production ready")
        
        # Execute standard fact-checking but filter for specific domain
        result = await self.v4_partial_fact_check(text)
        
        # Filter results to only include the specified domain
        domain_results = [r for r in result.fact_results if r["domain"] == domain]
        
        return V4PartialCoverageResult(
            original_text=result.original_text,
            enhanced_text=result.enhanced_text,
            fact_results=domain_results,
            domains_processed=[domain] if domain_results else [],
            domains_supported=[domain] if domain_results else [],
            domains_unsupported=[],
            processing_time=result.processing_time,
            production_status="DOMAIN_SPECIFIC",
            coverage_warnings=[],
            v4_confidence=result.v4_confidence
        )
    
    async def __aenter__(self):
        """Async context manager entry"""
        if not self.fact_checker:
            self.fact_checker = EnhancedFactCheckerV4()
            await self.fact_checker.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.fact_checker:
            await self.fact_checker.__aexit__(exc_type, exc_val, exc_tb)
    
    def run(self, host='0.0.0.0', port=8885, debug=False):
        """Run the Flask application"""
        self.logger.info(f"🚀 Starting Enhanced Fact-Checker V4 - Partial Production Service")
        self.logger.info(f"   Production Domains: {list(self.v4_production_domains.keys())}")
        self.logger.info(f"   Development Domains: {list(self.v5_development_domains.keys())}")
        self.logger.info(f"   Listening on {host}:{port}")
        
        self.app.run(host=host, port=port, debug=debug)

# Service instance
service = V4PartialFactCheckingService()

if __name__ == "__main__":
    service.run() 