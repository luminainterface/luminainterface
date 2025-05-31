#!/usr/bin/env python3
"""
CONFIDENCE-DRIVEN LORA CREATION SYSTEM DEPLOYMENT
================================================
Complete deployment script for the confidence-driven system that:
- Automatically detects when AI doesn't know something
- Creates targeted LoRAs to fill knowledge gaps
- Monitors confidence levels in real-time
- Triggers learning when uncertainty is detected
"""

import os
import sys
import time
import json
import asyncio
import aiohttp
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfidenceDrivenSystemDeployer:
    """Complete deployment manager for confidence-driven LoRA creation"""
    
    def __init__(self):
        self.project_dir = Path.cwd()
        self.required_files = [
            'confidence_driven_lora_creator.py',
            'ultimate_chat_orchestrator_with_confidence.py',
            'docker-compose-v10-ultimate.yml',
            'Dockerfile.confidence-driven-lora',
            'Dockerfile.ultimate-chat-confidence'
        ]
        
        self.services = {
            'confidence-driven-lora-creator': {
                'port': 8848,
                'capabilities': ['gap-detection', 'automatic-lora-creation', 'uncertainty-monitoring']
            },
            'ultimate-chat-orchestrator-with-confidence': {
                'port': 8950,
                'capabilities': ['multi-agent-coordination', 'confidence-monitoring', 'automatic-learning']
            },
            'enhanced-crawler-nlp': {
                'port': 8850,
                'capabilities': ['intelligent-crawling', 'nlp-filtering', 'content-quality-assessment']
            },
            'concept-training-worker': {
                'port': 8851,
                'capabilities': ['auto-lora-creation', 'quality-validation', 'domain-optimization']
            },
            'intelligent-retraining-coordinator': {
                'port': 8849,
                'capabilities': ['pipeline-orchestration', 'performance-monitoring', 'automated-triggers']
            }
        }
        
        self.confidence_features = [
            'Real-time confidence monitoring',
            'Automatic knowledge gap detection',
            'Self-triggered LoRA creation',
            'Intelligent uncertainty pattern recognition',
            'Demand-driven learning pipeline',
            'Transparent confidence reporting',
            'Knowledge gap prioritization',
            'End-to-end learning integration'
        ]
    
    def run_deployment(self):
        """Run complete deployment process"""
        
        print("🧠💡 CONFIDENCE-DRIVEN LORA CREATION SYSTEM DEPLOYMENT")
        print("🎯 Deploying automatic LoRA creation when AI says 'I don't know'")
        print("=" * 70)
        
        try:
            # Step 1: Prerequisites check
            print("\n1. 🔍 Checking Prerequisites...")
            self._check_prerequisites()
            
            # Step 2: File validation
            print("\n2. 📁 Validating Required Files...")
            self._validate_files()
            
            # Step 3: Create directories
            print("\n3. 📂 Creating Directories...")
            self._create_directories()
            
            # Step 4: Build Docker images
            print("\n4. 🐳 Building Docker Images...")
            self._build_docker_images()
            
            # Step 5: Deploy services
            print("\n5. 🚀 Deploying Services...")
            self._deploy_services()
            
            # Step 6: Wait for services
            print("\n6. ⏳ Waiting for Services to Start...")
            self._wait_for_services()
            
            # Step 7: Service health check
            print("\n7. 🏥 Health Check...")
            health_status = asyncio.run(self._check_service_health())
            
            # Step 8: Confidence system testing
            print("\n8. 🧠 Testing Confidence System...")
            test_results = asyncio.run(self._test_confidence_system())
            
            # Step 9: Integration validation
            print("\n9. 🔗 Validating Integration...")
            integration_results = asyncio.run(self._validate_integration())
            
            # Step 10: Generate deployment report
            print("\n10. 📊 Generating Deployment Report...")
            self._generate_deployment_report({
                'health_status': health_status,
                'test_results': test_results,
                'integration_results': integration_results
            })
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            print(f"❌ Deployment failed: {e}")
            sys.exit(1)
    
    def _check_prerequisites(self):
        """Check system prerequisites"""
        
        prerequisites = [
            ('docker', 'Docker is required for containerization'),
            ('docker-compose', 'Docker Compose is required for orchestration')
        ]
        
        for cmd, description in prerequisites:
            try:
                result = subprocess.run([cmd, '--version'], 
                                      capture_output=True, text=True, check=True)
                print(f"   ✅ {cmd}: {result.stdout.strip().split()[2]}")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print(f"   ❌ {cmd}: Not found - {description}")
                raise RuntimeError(f"{cmd} is not installed")
        
        # Check available ports
        print(f"   🔍 Checking port availability...")
        required_ports = [service['port'] for service in self.services.values()]
        for port in required_ports:
            if self._is_port_in_use(port):
                print(f"   ⚠️  Port {port} is in use")
            else:
                print(f"   ✅ Port {port} is available")
    
    def _validate_files(self):
        """Validate required files exist"""
        
        for file_path in self.required_files:
            if Path(file_path).exists():
                print(f"   ✅ {file_path}")
            else:
                print(f"   ❌ {file_path} - File missing")
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        # Check if requirements.txt exists or create it
        if not Path('requirements.txt').exists():
            print("   📝 Creating requirements.txt...")
            self._create_requirements_file()
            print("   ✅ requirements.txt created")
        else:
            print("   ✅ requirements.txt")
    
    def _create_directories(self):
        """Create necessary directories"""
        
        directories = [
            'logs/confidence_lora',
            'logs/chat_orchestrator',
            'conversations',
            'knowledge_gaps',
            'lora_models',
            'training_data'
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"   📁 Created: {dir_path}")
    
    def _create_requirements_file(self):
        """Create requirements.txt with necessary dependencies"""
        
        requirements = [
            "fastapi==0.104.1",
            "uvicorn==0.24.0",
            "aiohttp==3.9.1",
            "redis==5.0.1",
            "asyncio-mqtt==0.16.1",
            "pydantic==2.5.0",
            "transformers==4.36.2",
            "torch==2.1.1",
            "numpy==1.24.3",
            "scikit-learn==1.3.2",
            "arxiv==1.4.8",
            "feedparser==6.0.10",
            "beautifulsoup4==4.12.2",
            "requests==2.31.0",
            "nltk==3.8.1",
            "spacy==3.7.2",
            "sentence-transformers==2.2.2"
        ]
        
        with open('requirements.txt', 'w') as f:
            f.write('\n'.join(requirements))
    
    def _build_docker_images(self):
        """Build Docker images for confidence-driven services"""
        
        build_commands = [
            {
                'name': 'confidence-driven-lora-creator',
                'dockerfile': 'Dockerfile.confidence-driven-lora',
                'tag': 'confidence-driven-lora-creator:latest'
            },
            {
                'name': 'ultimate-chat-orchestrator-with-confidence',
                'dockerfile': 'Dockerfile.ultimate-chat-confidence',
                'tag': 'ultimate-chat-orchestrator-confidence:latest'
            }
        ]
        
        for build in build_commands:
            try:
                cmd = [
                    'docker', 'build',
                    '-f', build['dockerfile'],
                    '-t', build['tag'],
                    '.'
                ]
                
                print(f"   🔨 Building {build['name']}...")
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(f"   ✅ {build['name']}: Built successfully")
                
            except subprocess.CalledProcessError as e:
                print(f"   ❌ {build['name']}: Build failed")
                print(f"      Error: {e.stderr}")
                raise
    
    def _deploy_services(self):
        """Deploy services using docker-compose"""
        
        try:
            # Stop any existing services
            print("   🛑 Stopping existing services...")
            subprocess.run(['docker-compose', '-f', 'docker-compose-v10-ultimate.yml', 'down'], 
                         capture_output=True)
            
            # Start new services
            print("   🚀 Starting confidence-driven services...")
            result = subprocess.run([
                'docker-compose', '-f', 'docker-compose-v10-ultimate.yml',
                'up', '-d',
                'confidence-driven-lora-creator',
                'ultimate-chat-orchestrator-with-confidence',
                'enhanced-crawler-nlp',
                'concept-training-worker',
                'intelligent-retraining-coordinator'
            ], capture_output=True, text=True, check=True)
            
            print("   ✅ Services deployed successfully")
            
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Service deployment failed")
            print(f"      Error: {e.stderr}")
            raise
    
    def _wait_for_services(self):
        """Wait for services to become ready"""
        
        print("   ⏳ Waiting for services to start (60 seconds)...")
        time.sleep(60)
        
        # Additional wait for health checks
        print("   🏥 Allowing time for health checks (30 seconds)...")
        time.sleep(30)
    
    async def _check_service_health(self) -> Dict[str, Any]:
        """Check health of all deployed services"""
        
        health_status = {
            'total_services': len(self.services),
            'healthy_services': 0,
            'service_details': {}
        }
        
        for service_name, service_info in self.services.items():
            try:
                async with aiohttp.ClientSession() as session:
                    url = f"http://localhost:{service_info['port']}/health"
                    async with session.get(url, timeout=10) as response:
                        if response.status == 200:
                            data = await response.json()
                            health_status['service_details'][service_name] = {
                                'status': 'healthy',
                                'capabilities': data.get('capabilities', []),
                                'version': data.get('version', 'unknown')
                            }
                            health_status['healthy_services'] += 1
                            print(f"   ✅ {service_name}: Healthy")
                        else:
                            health_status['service_details'][service_name] = {
                                'status': 'unhealthy',
                                'error': f"HTTP {response.status}"
                            }
                            print(f"   ❌ {service_name}: Unhealthy (HTTP {response.status})")
            except Exception as e:
                health_status['service_details'][service_name] = {
                    'status': 'error',
                    'error': str(e)
                }
                print(f"   ⚠️  {service_name}: Error - {e}")
        
        return health_status
    
    async def _test_confidence_system(self) -> Dict[str, Any]:
        """Test confidence-driven functionality"""
        
        test_results = {
            'confidence_assessment_working': False,
            'uncertainty_detection_working': False,
            'lora_triggering_working': False,
            'chat_integration_working': False
        }
        
        try:
            # Test confidence assessment
            async with aiohttp.ClientSession() as session:
                # Test uncertainty detection
                async with session.post(
                    'http://localhost:8848/report_uncertainty',
                    json={
                        'query': 'What is the fictional XYZ-2024 algorithm?',
                        'response': "I don't have enough information about that.",
                        'context': {'test': 'deployment_validation'}
                    },
                    timeout=15
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        test_results['uncertainty_detection_working'] = True
                        test_results['lora_triggering_working'] = data.get('automatic_lora_triggered', False)
                        print(f"   🔍 Uncertainty detection: ✅ Working")
                        print(f"   🚀 LoRA triggering: {'✅ Working' if test_results['lora_triggering_working'] else '⚠️ Needs attention'}")
                
                # Test chat integration
                async with session.post(
                    'http://localhost:8950/chat_simple',
                    json={
                        'message': 'What is quantum computing?',
                        'user_id': 'deployment_test'
                    },
                    timeout=30
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        has_confidence = 'confidence_score' in data or 'confidence_level' in data
                        test_results['chat_integration_working'] = has_confidence
                        print(f"   💬 Chat integration: {'✅ Working' if has_confidence else '⚠️ Needs attention'}")
                
        except Exception as e:
            print(f"   ❌ Confidence testing failed: {e}")
        
        return test_results
    
    async def _validate_integration(self) -> Dict[str, Any]:
        """Validate end-to-end integration"""
        
        integration_results = {
            'pipeline_integration': False,
            'service_communication': False,
            'data_flow': False
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                # Test pipeline status
                async with session.get('http://localhost:8849/health', timeout=10) as response:
                    if response.status == 200:
                        integration_results['pipeline_integration'] = True
                        print(f"   🔄 Pipeline integration: ✅ Working")
                
                # Test service communication
                async with session.get('http://localhost:8848/confidence_analytics', timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        has_analytics = bool(data.get('learning_statistics'))
                        integration_results['service_communication'] = has_analytics
                        print(f"   📡 Service communication: {'✅ Working' if has_analytics else '⚠️ Limited'}")
                
                # Test data flow
                async with session.get('http://localhost:8848/knowledge_gaps', timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        has_data = 'knowledge_gaps' in data
                        integration_results['data_flow'] = has_data
                        print(f"   📊 Data flow: {'✅ Working' if has_data else '⚠️ Limited'}")
                
        except Exception as e:
            print(f"   ❌ Integration validation failed: {e}")
        
        return integration_results
    
    def _generate_deployment_report(self, results: Dict[str, Any]):
        """Generate comprehensive deployment report"""
        
        print("\n" + "=" * 70)
        print("🏆 CONFIDENCE-DRIVEN LORA CREATION SYSTEM DEPLOYMENT REPORT")
        print("=" * 70)
        
        # Service Health Summary
        health = results.get('health_status', {})
        healthy_services = health.get('healthy_services', 0)
        total_services = health.get('total_services', 0)
        health_percentage = (healthy_services / max(1, total_services)) * 100
        
        print(f"\n📊 SERVICE HEALTH:")
        print(f"   Services deployed: {total_services}")
        print(f"   Services healthy: {healthy_services}")
        print(f"   Health percentage: {health_percentage:.1f}%")
        
        # Confidence System Testing
        tests = results.get('test_results', {})
        working_features = sum(1 for feature in tests.values() if feature)
        total_features = len(tests)
        feature_percentage = (working_features / max(1, total_features)) * 100
        
        print(f"\n🧠 CONFIDENCE SYSTEM TESTING:")
        print(f"   Features tested: {total_features}")
        print(f"   Features working: {working_features}")
        print(f"   Feature success rate: {feature_percentage:.1f}%")
        
        # Integration Validation
        integration = results.get('integration_results', {})
        working_integrations = sum(1 for integration in integration.values() if integration)
        total_integrations = len(integration)
        integration_percentage = (working_integrations / max(1, total_integrations)) * 100
        
        print(f"\n🔗 INTEGRATION VALIDATION:")
        print(f"   Integrations tested: {total_integrations}")
        print(f"   Integrations working: {working_integrations}")
        print(f"   Integration success rate: {integration_percentage:.1f}%")
        
        # Overall System Status
        overall_success = (health_percentage + feature_percentage + integration_percentage) / 3
        
        print(f"\n🎯 OVERALL SYSTEM STATUS:")
        print(f"   Deployment success rate: {overall_success:.1f}%")
        
        if overall_success >= 80:
            print("   🎉 EXCELLENT: System deployed successfully!")
            print("   ✅ Confidence-driven LoRA creation is operational")
        elif overall_success >= 60:
            print("   ✅ GOOD: System mostly working with minor issues")
            print("   🔧 Some components may need attention")
        else:
            print("   ⚠️ NEEDS WORK: System has significant issues")
            print("   🛠️ Multiple components require attention")
        
        # Confidence-Driven Features Status
        print(f"\n🌟 CONFIDENCE-DRIVEN FEATURES:")
        for feature in self.confidence_features:
            status = "✅" if overall_success >= 70 else "⚠️"
            print(f"   {status} {feature}")
        
        # Usage Instructions
        if overall_success >= 70:
            print(f"\n🚀 SYSTEM READY FOR USE!")
            print(f"   The confidence-driven system will automatically:")
            print(f"   • Monitor AI confidence levels in real-time")
            print(f"   • Detect when the AI says 'I don't know'")
            print(f"   • Trigger automatic LoRA creation for knowledge gaps")
            print(f"   • Improve responses through targeted learning")
            
            print(f"\n💬 CHAT ENDPOINT:")
            print(f"   URL: http://localhost:8950/chat_simple")
            print(f"   Features: Confidence monitoring, automatic learning")
            
            print(f"\n📊 CONFIDENCE MONITORING:")
            print(f"   URL: http://localhost:8848/confidence_analytics")
            print(f"   Features: Real-time gap tracking, LoRA status")
            
            print(f"\n🔄 RETRAINING PIPELINE:")
            print(f"   URL: http://localhost:8849/performance_dashboard")
            print(f"   Features: Performance monitoring, automated triggers")
        
        print(f"\n📋 DEPLOYMENT SUMMARY:")
        print(f"   Total services deployed: {total_services}")
        print(f"   Healthy services: {healthy_services}")
        print(f"   Confidence system working: {'Yes' if feature_percentage >= 70 else 'Partial'}")
        print(f"   Integration status: {'Complete' if integration_percentage >= 70 else 'Partial'}")
        print(f"   Ready for production: {'Yes' if overall_success >= 80 else 'Needs work'}")
        
        print("\n" + "=" * 70)
        
        # Save deployment report
        report_data = {
            'deployment_time': datetime.now().isoformat(),
            'overall_success_rate': overall_success,
            'service_health': health,
            'confidence_testing': tests,
            'integration_results': integration,
            'features_status': self.confidence_features
        }
        
        with open('deployment_report.json', 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"📄 Deployment report saved to: deployment_report.json")
    
    @staticmethod
    def _is_port_in_use(port: int) -> bool:
        """Check if a port is in use"""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

def main():
    """Main deployment function"""
    
    deployer = ConfidenceDrivenSystemDeployer()
    deployer.run_deployment()

if __name__ == "__main__":
    main() 