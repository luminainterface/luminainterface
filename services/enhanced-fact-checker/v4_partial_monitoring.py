#!/usr/bin/env python3
"""
V4 PARTIAL DEPLOYMENT MONITORING SERVICE
=======================================

REAL-TIME MONITORING FOR V4 PRODUCTION DOMAINS:
✅ Technology, Medicine, Psychology, Geography

DEVELOPMENT TRACKING FOR V5 DOMAINS:
🚧 Chemistry, Engineering, Sports, Global Issues, Edge Cases

METRICS TRACKED:
- Domain coverage rates
- Processing performance
- Error detection effectiveness
- Production readiness indicators
"""

import asyncio
import aiohttp
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

@dataclass
class V4DomainMetrics:
    """V4 domain-specific metrics"""
    domain: str
    requests_count: int
    processing_time_avg: float
    error_detection_rate: float
    confidence_avg: float
    production_status: str
    last_24h_requests: int
    success_rate: float

@dataclass
class V4SystemMetrics:
    """V4 system-wide metrics"""
    total_requests: int
    production_domain_requests: int
    development_domain_requests: int
    avg_processing_time: float
    production_coverage_rate: float
    uptime_seconds: float
    error_rate: float
    deployment_status: str

class V4PartialDeploymentMonitor:
    """Monitor V4 partial deployment performance and coverage"""
    
    def __init__(self, fact_checker_url="http://localhost:8885"):
        self.fact_checker_url = fact_checker_url
        self.logger = self._setup_logging()
        
        # V4 Production domains
        self.production_domains = {"TECHNOLOGY", "MEDICINE", "PSYCHOLOGY", "GEOGRAPHY"}
        self.development_domains = {"CHEMISTRY", "ENGINEERING", "SPORTS", "GLOBAL_ISSUES", "EDGE_CASES"}
        
        # Metrics storage
        self.domain_metrics = {}
        self.request_history = deque(maxlen=1000)  # Last 1000 requests
        self.start_time = time.time()
        
        # Initialize domain metrics
        for domain in self.production_domains | self.development_domains:
            self.domain_metrics[domain] = {
                "requests": 0,
                "processing_times": deque(maxlen=100),
                "error_detections": deque(maxlen=100),
                "confidences": deque(maxlen=100),
                "successes": deque(maxlen=100),
                "last_24h": deque(maxlen=2400)  # 24h * 100 requests/hour max
            }
    
    def _setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - V4Monitor - %(levelname)s - %(message)s'
        )
        return logging.getLogger("V4PartialMonitor")
    
    async def check_service_health(self) -> Dict[str, Any]:
        """Check V4 fact-checker service health"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.fact_checker_url}/health") as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {"status": "unhealthy", "error": f"HTTP {response.status}"}
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def get_coverage_status(self) -> Dict[str, Any]:
        """Get current coverage status"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.fact_checker_url}/coverage") as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {"error": f"HTTP {response.status}"}
        except Exception as e:
            self.logger.error(f"Coverage check failed: {str(e)}")
            return {"error": str(e)}
    
    async def test_domain_performance(self, domain: str) -> Dict[str, Any]:
        """Test performance for a specific domain"""
        
        # Domain-specific test cases
        test_cases = {
            "TECHNOLOGY": "Java was developed by Microsoft. Python was created by Guido van Rossum.",
            "MEDICINE": "The human heart has four chambers. Antibiotics are effective against viral infections.",
            "PSYCHOLOGY": "All behavior is learned through conditioning. Pavlov studied classical conditioning.",
            "GEOGRAPHY": "Australia's capital is Sydney. The G7 includes Australia as a member.",
            "CHEMISTRY": "Water has a bond angle of 109.5 degrees. DNA is triple-helix.",
            "ENGINEERING": "All conductors have zero resistance. Ohm's law relates voltage and current.",
            "SPORTS": "The Olympics occur every 2 years. Basketball teams have 6 players on court.",
            "GLOBAL_ISSUES": "Climate change is entirely natural. All vaccines cause autism.",
            "EDGE_CASES": "Water boils at 100°C everywhere. Nothing travels faster than light anywhere."
        }
        
        if domain not in test_cases:
            return {"error": f"No test case for domain {domain}"}
        
        test_text = test_cases[domain]
        
        try:
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.fact_checker_url}/fact-check",
                    json={"text": test_text}
                ) as response:
                    processing_time = time.time() - start_time
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        # Record metrics
                        self._record_domain_metrics(domain, result, processing_time, True)
                        
                        return {
                            "domain": domain,
                            "processing_time": processing_time,
                            "production_supported": domain in self.production_domains,
                            "result": result,
                            "test_passed": True
                        }
                    else:
                        self._record_domain_metrics(domain, {}, processing_time, False)
                        return {
                            "domain": domain,
                            "processing_time": processing_time,
                            "error": f"HTTP {response.status}",
                            "test_passed": False
                        }
                        
        except Exception as e:
            self.logger.error(f"Domain test failed for {domain}: {str(e)}")
            return {
                "domain": domain,
                "error": str(e),
                "test_passed": False
            }
    
    def _record_domain_metrics(self, domain: str, result: Dict, processing_time: float, success: bool):
        """Record metrics for a domain"""
        
        if domain not in self.domain_metrics:
            return
        
        metrics = self.domain_metrics[domain]
        metrics["requests"] += 1
        metrics["processing_times"].append(processing_time)
        metrics["successes"].append(success)
        metrics["last_24h"].append(time.time())
        
        if success and "fact_results" in result:
            # Calculate error detection rate
            fact_results = result["fact_results"]
            errors_found = sum(1 for r in fact_results if not r.get("is_accurate", True))
            total_claims = len(fact_results)
            error_detection_rate = errors_found / max(total_claims, 1)
            
            # Calculate average confidence
            confidences = [r.get("confidence_score", 0.5) for r in fact_results]
            avg_confidence = sum(confidences) / max(len(confidences), 1)
            
            metrics["error_detections"].append(error_detection_rate)
            metrics["confidences"].append(avg_confidence)
        
        # Record in request history
        self.request_history.append({
            "timestamp": time.time(),
            "domain": domain,
            "processing_time": processing_time,
            "success": success,
            "production_domain": domain in self.production_domains
        })
    
    def get_domain_metrics(self, domain: str) -> V4DomainMetrics:
        """Get metrics for a specific domain"""
        
        if domain not in self.domain_metrics:
            return V4DomainMetrics(
                domain=domain,
                requests_count=0,
                processing_time_avg=0.0,
                error_detection_rate=0.0,
                confidence_avg=0.0,
                production_status="UNKNOWN",
                last_24h_requests=0,
                success_rate=0.0
            )
        
        metrics = self.domain_metrics[domain]
        
        # Calculate averages
        processing_times = list(metrics["processing_times"])
        error_detections = list(metrics["error_detections"])
        confidences = list(metrics["confidences"])
        successes = list(metrics["successes"])
        
        avg_processing_time = sum(processing_times) / max(len(processing_times), 1)
        avg_error_detection = sum(error_detections) / max(len(error_detections), 1)
        avg_confidence = sum(confidences) / max(len(confidences), 1)
        success_rate = sum(successes) / max(len(successes), 1)
        
        # Count last 24h requests
        cutoff_time = time.time() - 86400  # 24 hours
        last_24h_requests = sum(1 for t in metrics["last_24h"] if t > cutoff_time)
        
        # Determine production status
        if domain in self.production_domains:
            production_status = "PRODUCTION_READY"
        elif domain in self.development_domains:
            production_status = "V5_DEVELOPMENT"
        else:
            production_status = "UNKNOWN"
        
        return V4DomainMetrics(
            domain=domain,
            requests_count=metrics["requests"],
            processing_time_avg=avg_processing_time,
            error_detection_rate=avg_error_detection,
            confidence_avg=avg_confidence,
            production_status=production_status,
            last_24h_requests=last_24h_requests,
            success_rate=success_rate
        )
    
    def get_system_metrics(self) -> V4SystemMetrics:
        """Get overall system metrics"""
        
        total_requests = len(self.request_history)
        production_requests = sum(1 for r in self.request_history if r["production_domain"])
        development_requests = total_requests - production_requests
        
        if total_requests > 0:
            avg_processing_time = sum(r["processing_time"] for r in self.request_history) / total_requests
            success_rate = sum(1 for r in self.request_history if r["success"]) / total_requests
            production_coverage_rate = production_requests / total_requests
        else:
            avg_processing_time = 0.0
            success_rate = 1.0
            production_coverage_rate = 0.0
        
        uptime_seconds = time.time() - self.start_time
        error_rate = 1.0 - success_rate
        
        # Determine deployment status
        if production_coverage_rate > 0.8 and success_rate > 0.9 and avg_processing_time < 0.1:
            deployment_status = "EXCELLENT"
        elif production_coverage_rate > 0.6 and success_rate > 0.8:
            deployment_status = "GOOD"
        elif production_coverage_rate > 0.4:
            deployment_status = "ACCEPTABLE"
        else:
            deployment_status = "NEEDS_IMPROVEMENT"
        
        return V4SystemMetrics(
            total_requests=total_requests,
            production_domain_requests=production_requests,
            development_domain_requests=development_requests,
            avg_processing_time=avg_processing_time,
            production_coverage_rate=production_coverage_rate,
            uptime_seconds=uptime_seconds,
            error_rate=error_rate,
            deployment_status=deployment_status
        )
    
    async def run_comprehensive_monitoring(self) -> Dict[str, Any]:
        """Run comprehensive monitoring of V4 partial deployment"""
        
        self.logger.info("🔍 Starting V4 Partial Deployment Comprehensive Monitoring")
        
        # Check service health
        health_status = await self.check_service_health()
        self.logger.info(f"Service Health: {health_status.get('status', 'unknown')}")
        
        # Get coverage status
        coverage_status = await self.get_coverage_status()
        self.logger.info(f"Coverage Status: {coverage_status}")
        
        # Test all domains
        domain_results = {}
        for domain in self.production_domains | self.development_domains:
            self.logger.info(f"Testing domain: {domain}")
            result = await self.test_domain_performance(domain)
            domain_results[domain] = result
            
            if result.get("test_passed"):
                self.logger.info(f"✅ {domain}: {result['processing_time']:.3f}s")
            else:
                self.logger.warning(f"❌ {domain}: {result.get('error', 'Failed')}")
        
        # Generate metrics
        domain_metrics = {}
        for domain in self.production_domains | self.development_domains:
            domain_metrics[domain] = asdict(self.get_domain_metrics(domain))
        
        system_metrics = asdict(self.get_system_metrics())
        
        # Generate report
        report = {
            "timestamp": datetime.now().isoformat(),
            "service_health": health_status,
            "coverage_status": coverage_status,
            "domain_test_results": domain_results,
            "domain_metrics": domain_metrics,
            "system_metrics": system_metrics,
            "production_domains": list(self.production_domains),
            "development_domains": list(self.development_domains),
            "v4_deployment_assessment": self._generate_deployment_assessment(domain_results, system_metrics)
        }
        
        self.logger.info("📊 V4 Monitoring Complete")
        return report
    
    def _generate_deployment_assessment(self, domain_results: Dict, system_metrics: Dict) -> Dict[str, Any]:
        """Generate V4 deployment assessment"""
        
        production_domains_working = sum(
            1 for domain in self.production_domains 
            if domain_results.get(domain, {}).get("test_passed", False)
        )
        
        development_domains_working = sum(
            1 for domain in self.development_domains 
            if domain_results.get(domain, {}).get("test_passed", False)
        )
        
        production_success_rate = production_domains_working / len(self.production_domains)
        development_success_rate = development_domains_working / max(len(self.development_domains), 1)
        
        avg_production_time = sum(
            domain_results.get(domain, {}).get("processing_time", 0)
            for domain in self.production_domains
            if domain_results.get(domain, {}).get("test_passed", False)
        ) / max(production_domains_working, 1)
        
        # Assessment
        if production_success_rate >= 1.0 and avg_production_time < 0.05:
            assessment = "EXCELLENT_PRODUCTION_READY"
            recommendation = "V4 partial deployment performing excellently"
        elif production_success_rate >= 0.75:
            assessment = "GOOD_PRODUCTION_READY"
            recommendation = "V4 partial deployment performing well"
        elif production_success_rate >= 0.5:
            assessment = "ACCEPTABLE_WITH_ISSUES"
            recommendation = "V4 has issues but partially functional"
        else:
            assessment = "NEEDS_IMMEDIATE_ATTENTION"
            recommendation = "V4 deployment requires immediate fixes"
        
        return {
            "assessment": assessment,
            "recommendation": recommendation,
            "production_success_rate": production_success_rate,
            "development_success_rate": development_success_rate,
            "avg_production_processing_time": avg_production_time,
            "production_domains_working": production_domains_working,
            "total_production_domains": len(self.production_domains),
            "v5_development_status": f"{development_domains_working}/{len(self.development_domains)} domains responding"
        }

async def main():
    """Run V4 monitoring"""
    monitor = V4PartialDeploymentMonitor()
    report = await monitor.run_comprehensive_monitoring()
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"v4_partial_deployment_monitoring_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 V4 PARTIAL DEPLOYMENT MONITORING REPORT")
    print(f"=" * 60)
    print(f"Service Health: {report['service_health']['status']}")
    print(f"Production Domains: {report['v4_deployment_assessment']['production_domains_working']}/{len(report['production_domains'])}")
    print(f"Assessment: {report['v4_deployment_assessment']['assessment']}")
    print(f"Recommendation: {report['v4_deployment_assessment']['recommendation']}")
    print(f"Report saved: {filename}")

if __name__ == "__main__":
    asyncio.run(main()) 