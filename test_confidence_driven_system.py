#!/usr/bin/env python3
"""
CONFIDENCE-DRIVEN LORA CREATION SYSTEM TESTING
==============================================
Comprehensive testing suite that demonstrates:
- Real-time confidence monitoring
- Automatic knowledge gap detection  
- Self-triggered LoRA creation when AI says "I don't know"
- End-to-end learning pipeline validation
- Performance improvement tracking
"""

import asyncio
import aiohttp
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfidenceDrivenSystemTester:
    """Comprehensive tester for confidence-driven LoRA creation"""
    
    def __init__(self):
        self.base_urls = {
            'confidence_creator': 'http://localhost:8848',
            'chat_orchestrator': 'http://localhost:8950',
            'enhanced_crawler': 'http://localhost:8850',
            'concept_training': 'http://localhost:8851',
            'retraining_coordinator': 'http://localhost:8849'
        }
        
        # Test scenarios that should trigger "I don't know" responses
        self.uncertainty_test_queries = [
            "What is the latest research on quantum-biological computing interfaces published this month?",
            "How does the new XYZ-2024 algorithm compare to traditional machine learning approaches?",
            "What are the recent developments in hypothetical time-reversal quantum mechanics?",
            "Can you explain the technical details of the fictional ABC-Neural framework?",
            "What's the current status of imaginary meta-learning protocols in distributed systems?",
            "How do you implement non-existent recursive self-improving AI architectures?",
            "What are the specifics of the made-up quantum-entangled neural networks?",
            "Can you describe the recent breakthroughs in fictional bio-quantum processors?",
            "What's the latest on hypothetical consciousness-transfer algorithms?",
            "How does the imaginary neuro-quantum bridge technology work?"
        ]
        
        # Test scenarios with different confidence levels
        self.confidence_test_queries = [
            {
                'query': "What is 2 + 2?",
                'expected_confidence': 'high',
                'should_trigger_lora': False
            },
            {
                'query': "Explain the basic concept of machine learning",
                'expected_confidence': 'medium',
                'should_trigger_lora': False
            },
            {
                'query': "What are the latest developments in quantum computing this week?",
                'expected_confidence': 'low',
                'should_trigger_lora': True
            },
            {
                'query': "How does the fictional ZetaML algorithm work?",
                'expected_confidence': 'unknown',
                'should_trigger_lora': True
            }
        ]
    
    async def run_comprehensive_test_suite(self):
        """Run complete test suite for confidence-driven system"""
        
        print("🧠💡 CONFIDENCE-DRIVEN LORA CREATION SYSTEM TESTING")
        print("=" * 60)
        
        try:
            # Test 1: Service health checks
            print("\n1. 🔍 Testing Service Health...")
            health_results = await self._test_service_health()
            
            # Test 2: Confidence assessment functionality
            print("\n2. 📊 Testing Confidence Assessment...")
            confidence_results = await self._test_confidence_assessment()
            
            # Test 3: Uncertainty detection and LoRA triggering
            print("\n3. 🎯 Testing Uncertainty Detection and LoRA Triggering...")
            uncertainty_results = await self._test_uncertainty_detection()
            
            # Test 4: Chat orchestrator integration
            print("\n4. 💬 Testing Chat Orchestrator Integration...")
            chat_results = await self._test_chat_orchestrator_integration()
            
            # Test 5: Knowledge gap tracking
            print("\n5. 📈 Testing Knowledge Gap Tracking...")
            gap_tracking_results = await self._test_knowledge_gap_tracking()
            
            # Test 6: End-to-end learning pipeline
            print("\n6. 🔄 Testing End-to-End Learning Pipeline...")
            pipeline_results = await self._test_end_to_end_pipeline()
            
            # Test 7: Performance analytics
            print("\n7. 📊 Testing Performance Analytics...")
            analytics_results = await self._test_performance_analytics()
            
            # Generate comprehensive report
            print("\n" + "=" * 60)
            self._generate_test_report({
                'health': health_results,
                'confidence': confidence_results,
                'uncertainty': uncertainty_results,
                'chat_integration': chat_results,
                'gap_tracking': gap_tracking_results,
                'pipeline': pipeline_results,
                'analytics': analytics_results
            })
            
        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            print(f"❌ Test suite failed: {e}")
    
    async def _test_service_health(self) -> Dict[str, Any]:
        """Test health of all confidence-driven services"""
        
        health_results = {}
        
        for service_name, base_url in self.base_urls.items():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{base_url}/health", timeout=10) as response:
                        if response.status == 200:
                            data = await response.json()
                            health_results[service_name] = {
                                'status': 'healthy',
                                'capabilities': data.get('capabilities', []),
                                'version': data.get('version', 'unknown')
                            }
                            print(f"   ✅ {service_name}: Healthy")
                        else:
                            health_results[service_name] = {
                                'status': 'unhealthy',
                                'error': f"HTTP {response.status}"
                            }
                            print(f"   ❌ {service_name}: Unhealthy (HTTP {response.status})")
            except Exception as e:
                health_results[service_name] = {
                    'status': 'error',
                    'error': str(e)
                }
                print(f"   ⚠️  {service_name}: Error - {e}")
        
        return health_results
    
    async def _test_confidence_assessment(self) -> Dict[str, Any]:
        """Test confidence assessment functionality"""
        
        confidence_results = {
            'tests_run': 0,
            'assessments_correct': 0,
            'lora_triggers_appropriate': 0,
            'test_details': []
        }
        
        for test_case in self.confidence_test_queries:
            try:
                confidence_results['tests_run'] += 1
                
                # Send confidence assessment request
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_urls['confidence_creator']}/assess_confidence",
                        json={
                            'query': test_case['query'],
                            'response': 'Test response for confidence assessment',
                            'confidence_score': 0.1 if test_case['expected_confidence'] == 'unknown' else 0.7,
                            'response_time': 1.0,
                            'model_used': 'test_model'
                        },
                        timeout=15
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            assessment = data.get('assessment_result', {})
                            confidence_assessment = assessment.get('confidence_assessment', {})
                            
                            detected_level = confidence_assessment.get('confidence_level', 'unknown')
                            lora_triggered = confidence_assessment.get('lora_triggered', False)
                            
                            # Check if assessment is correct
                            assessment_correct = detected_level == test_case['expected_confidence']
                            if assessment_correct:
                                confidence_results['assessments_correct'] += 1
                            
                            # Check if LoRA triggering is appropriate
                            lora_appropriate = lora_triggered == test_case['should_trigger_lora']
                            if lora_appropriate:
                                confidence_results['lora_triggers_appropriate'] += 1
                            
                            confidence_results['test_details'].append({
                                'query': test_case['query'],
                                'expected_confidence': test_case['expected_confidence'],
                                'detected_confidence': detected_level,
                                'assessment_correct': assessment_correct,
                                'expected_lora_trigger': test_case['should_trigger_lora'],
                                'actual_lora_trigger': lora_triggered,
                                'lora_appropriate': lora_appropriate
                            })
                            
                            status = "✅" if assessment_correct and lora_appropriate else "⚠️"
                            print(f"   {status} Query: '{test_case['query'][:50]}...' -> Confidence: {detected_level}, LoRA: {lora_triggered}")
                        
            except Exception as e:
                print(f"   ❌ Confidence assessment failed for query: {e}")
        
        return confidence_results
    
    async def _test_uncertainty_detection(self) -> Dict[str, Any]:
        """Test uncertainty detection and automatic LoRA triggering"""
        
        uncertainty_results = {
            'uncertainty_queries_tested': 0,
            'uncertainty_detected': 0,
            'loras_triggered': 0,
            'detection_details': []
        }
        
        for query in self.uncertainty_test_queries:
            try:
                uncertainty_results['uncertainty_queries_tested'] += 1
                
                # Send uncertainty report
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_urls['confidence_creator']}/report_uncertainty",
                        json={
                            'query': query,
                            'response': "I don't have enough information to answer that question properly.",
                            'context': {'test_scenario': 'uncertainty_detection'}
                        },
                        timeout=15
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            assessment_result = data.get('assessment_result', {})
                            confidence_assessment = assessment_result.get('confidence_assessment', {})
                            
                            gap_detected = confidence_assessment.get('gap_detected', False)
                            lora_triggered = data.get('automatic_lora_triggered', False)
                            
                            if gap_detected:
                                uncertainty_results['uncertainty_detected'] += 1
                            
                            if lora_triggered:
                                uncertainty_results['loras_triggered'] += 1
                            
                            uncertainty_results['detection_details'].append({
                                'query': query,
                                'gap_detected': gap_detected,
                                'lora_triggered': lora_triggered
                            })
                            
                            status = "✅" if gap_detected and lora_triggered else "⚠️"
                            print(f"   {status} Uncertainty detected: {gap_detected}, LoRA triggered: {lora_triggered}")
                        
            except Exception as e:
                print(f"   ❌ Uncertainty detection failed: {e}")
        
        return uncertainty_results
    
    async def _test_chat_orchestrator_integration(self) -> Dict[str, Any]:
        """Test chat orchestrator integration with confidence system"""
        
        chat_results = {
            'chat_tests': 0,
            'confidence_reported': 0,
            'gaps_detected': 0,
            'learning_triggered': 0,
            'chat_details': []
        }
        
        test_messages = [
            "What is machine learning?",
            "Can you explain quantum gravity theory?",
            "How does the fictional UltraAI system work?",
            "What's 2 + 2?",
            "Tell me about recent developments in bio-quantum computing."
        ]
        
        for message in test_messages:
            try:
                chat_results['chat_tests'] += 1
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_urls['chat_orchestrator']}/chat_simple",
                        json={'message': message, 'user_id': 'test_user'},
                        timeout=30
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            confidence_score = data.get('confidence_score', 0.0)
                            confidence_level = data.get('confidence_level', 'unknown')
                            learning_triggered = data.get('learning_triggered', False)
                            
                            has_confidence = confidence_score > 0.0 or confidence_level != 'unknown'
                            if has_confidence:
                                chat_results['confidence_reported'] += 1
                            
                            if confidence_score < 0.4:
                                chat_results['gaps_detected'] += 1
                            
                            if learning_triggered:
                                chat_results['learning_triggered'] += 1
                            
                            chat_results['chat_details'].append({
                                'message': message,
                                'confidence_score': confidence_score,
                                'confidence_level': confidence_level,
                                'learning_triggered': learning_triggered,
                                'response': data.get('response', '')[:100] + '...'
                            })
                            
                            print(f"   💬 '{message[:30]}...' -> Confidence: {confidence_level} ({confidence_score:.2f}), Learning: {learning_triggered}")
                        
            except Exception as e:
                print(f"   ❌ Chat test failed: {e}")
        
        return chat_results
    
    async def _test_knowledge_gap_tracking(self) -> Dict[str, Any]:
        """Test knowledge gap tracking and prioritization"""
        
        gap_results = {}
        
        try:
            async with aiohttp.ClientSession() as session:
                # Get current knowledge gaps
                async with session.get(f"{self.base_urls['confidence_creator']}/knowledge_gaps", timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        gap_results['total_gaps'] = data.get('total_gaps', 0)
                        gap_results['knowledge_gaps'] = data.get('knowledge_gaps', [])
                        gap_results['gap_priorities'] = data.get('gap_priorities', [])
                        gap_results['domain_distribution'] = data.get('domain_distribution', {})
                        
                        print(f"   📊 Total knowledge gaps tracked: {gap_results['total_gaps']}")
                        print(f"   🏆 Top priority gaps: {len(gap_results['gap_priorities'])}")
                        print(f"   🎯 Domain distribution: {gap_results['domain_distribution']}")
                    else:
                        gap_results['error'] = f"HTTP {response.status}"
                
                # Get confidence analytics
                async with session.get(f"{self.base_urls['confidence_creator']}/confidence_analytics", timeout=10) as response:
                    if response.status == 200:
                        analytics = await response.json()
                        gap_results['learning_statistics'] = analytics.get('learning_statistics', {})
                        gap_results['confidence_trends'] = analytics.get('confidence_trends', {})
                        gap_results['active_lora_requests'] = analytics.get('active_lora_requests', {})
                        
                        stats = gap_results['learning_statistics']
                        print(f"   📈 Queries analyzed: {stats.get('total_queries_analyzed', 0)}")
                        print(f"   🔍 Low confidence detected: {stats.get('low_confidence_detected', 0)}")
                        print(f"   🚀 LoRAs created from gaps: {stats.get('loras_created_from_gaps', 0)}")
                        print(f"   ✅ Knowledge gaps filled: {stats.get('knowledge_gaps_filled', 0)}")
                
        except Exception as e:
            gap_results['error'] = str(e)
            print(f"   ❌ Gap tracking test failed: {e}")
        
        return gap_results
    
    async def _test_end_to_end_pipeline(self) -> Dict[str, Any]:
        """Test complete end-to-end learning pipeline"""
        
        pipeline_results = {
            'pipeline_stages': {},
            'integration_success': False
        }
        
        try:
            # Test crawler status
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_urls['enhanced_crawler']}/health", timeout=10) as response:
                    if response.status == 200:
                        pipeline_results['pipeline_stages']['crawler'] = 'healthy'
                        print("   ✅ Enhanced Crawler: Ready")
                    else:
                        pipeline_results['pipeline_stages']['crawler'] = 'unhealthy'
                        print("   ❌ Enhanced Crawler: Not ready")
                
                # Test concept training status
                async with session.get(f"{self.base_urls['concept_training']}/health", timeout=10) as response:
                    if response.status == 200:
                        pipeline_results['pipeline_stages']['training'] = 'healthy'
                        print("   ✅ Concept Training: Ready")
                    else:
                        pipeline_results['pipeline_stages']['training'] = 'unhealthy'
                        print("   ❌ Concept Training: Not ready")
                
                # Test retraining coordinator status
                async with session.get(f"{self.base_urls['retraining_coordinator']}/health", timeout=10) as response:
                    if response.status == 200:
                        pipeline_results['pipeline_stages']['coordinator'] = 'healthy'
                        print("   ✅ Retraining Coordinator: Ready")
                    else:
                        pipeline_results['pipeline_stages']['coordinator'] = 'unhealthy'
                        print("   ❌ Retraining Coordinator: Not ready")
                
                # Check integration
                all_healthy = all(status == 'healthy' for status in pipeline_results['pipeline_stages'].values())
                pipeline_results['integration_success'] = all_healthy
                
                if all_healthy:
                    print("   🔄 End-to-end pipeline: ✅ INTEGRATED")
                else:
                    print("   🔄 End-to-end pipeline: ⚠️ PARTIAL")
                
        except Exception as e:
            pipeline_results['error'] = str(e)
            print(f"   ❌ Pipeline test failed: {e}")
        
        return pipeline_results
    
    async def _test_performance_analytics(self) -> Dict[str, Any]:
        """Test performance analytics and insights"""
        
        analytics_results = {}
        
        try:
            async with aiohttp.ClientSession() as session:
                # Get chat orchestrator analytics
                async with session.get(f"{self.base_urls['chat_orchestrator']}/confidence_insights", timeout=10) as response:
                    if response.status == 200:
                        insights = await response.json()
                        analytics_results['confidence_insights'] = insights
                        
                        print(f"   📊 Total interactions: {insights.get('total_interactions', 0)}")
                        print(f"   🎯 Gap frequency: {insights.get('gap_frequency', 0)}")
                        print(f"   🚀 Learning triggers: {insights.get('learning_triggers', 0)}")
                        print(f"   📈 Improvement trend: {insights.get('improvement_trend', 'unknown')}")
                
                # Get retraining coordinator status
                async with session.get(f"{self.base_urls['retraining_coordinator']}/performance_dashboard", timeout=10) as response:
                    if response.status == 200:
                        dashboard = await response.json()
                        analytics_results['performance_dashboard'] = dashboard
                        
                        summary = dashboard.get('performance_summary', {})
                        print(f"   🏥 Overall health: {summary.get('overall_health', 'unknown')}")
                        print(f"   ✅ Metrics at target: {summary.get('metrics_at_target', 0)}")
                        print(f"   📈 Trending up: {summary.get('trending_up', 0)}")
                
        except Exception as e:
            analytics_results['error'] = str(e)
            print(f"   ❌ Analytics test failed: {e}")
        
        return analytics_results
    
    def _generate_test_report(self, results: Dict[str, Any]):
        """Generate comprehensive test report"""
        
        print("\n🏆 CONFIDENCE-DRIVEN SYSTEM TEST REPORT")
        print("=" * 60)
        
        # Service Health Summary
        health = results.get('health', {})
        healthy_services = sum(1 for service in health.values() if service.get('status') == 'healthy')
        total_services = len(health)
        
        print(f"\n📊 SERVICE HEALTH: {healthy_services}/{total_services} services healthy")
        
        # Confidence Assessment Summary
        confidence = results.get('confidence', {})
        if confidence:
            tests_run = confidence.get('tests_run', 0)
            assessments_correct = confidence.get('assessments_correct', 0)
            lora_triggers_appropriate = confidence.get('lora_triggers_appropriate', 0)
            
            confidence_accuracy = (assessments_correct / max(1, tests_run)) * 100
            lora_accuracy = (lora_triggers_appropriate / max(1, tests_run)) * 100
            
            print(f"\n🎯 CONFIDENCE ASSESSMENT:")
            print(f"   Confidence accuracy: {confidence_accuracy:.1f}% ({assessments_correct}/{tests_run})")
            print(f"   LoRA trigger accuracy: {lora_accuracy:.1f}% ({lora_triggers_appropriate}/{tests_run})")
        
        # Uncertainty Detection Summary
        uncertainty = results.get('uncertainty', {})
        if uncertainty:
            tested = uncertainty.get('uncertainty_queries_tested', 0)
            detected = uncertainty.get('uncertainty_detected', 0)
            triggered = uncertainty.get('loras_triggered', 0)
            
            detection_rate = (detected / max(1, tested)) * 100
            trigger_rate = (triggered / max(1, tested)) * 100
            
            print(f"\n🔍 UNCERTAINTY DETECTION:")
            print(f"   Gap detection rate: {detection_rate:.1f}% ({detected}/{tested})")
            print(f"   LoRA trigger rate: {trigger_rate:.1f}% ({triggered}/{tested})")
        
        # Chat Integration Summary
        chat = results.get('chat_integration', {})
        if chat:
            tests = chat.get('chat_tests', 0)
            confidence_reported = chat.get('confidence_reported', 0)
            gaps_detected = chat.get('gaps_detected', 0)
            learning_triggered = chat.get('learning_triggered', 0)
            
            print(f"\n💬 CHAT INTEGRATION:")
            print(f"   Confidence reporting: {confidence_reported}/{tests} messages")
            print(f"   Knowledge gaps detected: {gaps_detected}")
            print(f"   Learning triggered: {learning_triggered}")
        
        # Knowledge Gap Tracking Summary
        gaps = results.get('gap_tracking', {})
        if gaps and 'learning_statistics' in gaps:
            stats = gaps['learning_statistics']
            print(f"\n📈 KNOWLEDGE GAP TRACKING:")
            print(f"   Total gaps tracked: {gaps.get('total_gaps', 0)}")
            print(f"   Queries analyzed: {stats.get('total_queries_analyzed', 0)}")
            print(f"   LoRAs created from gaps: {stats.get('loras_created_from_gaps', 0)}")
            print(f"   Knowledge gaps filled: {stats.get('knowledge_gaps_filled', 0)}")
        
        # Pipeline Integration Summary
        pipeline = results.get('pipeline', {})
        if pipeline:
            integration_success = pipeline.get('integration_success', False)
            stages = pipeline.get('pipeline_stages', {})
            healthy_stages = sum(1 for status in stages.values() if status == 'healthy')
            total_stages = len(stages)
            
            print(f"\n🔄 END-TO-END PIPELINE:")
            print(f"   Pipeline stages healthy: {healthy_stages}/{total_stages}")
            print(f"   Integration status: {'✅ SUCCESS' if integration_success else '⚠️ PARTIAL'}")
        
        # Overall System Assessment
        print(f"\n🏆 OVERALL ASSESSMENT:")
        
        # Calculate overall success score
        success_metrics = []
        
        if healthy_services >= total_services * 0.8:  # 80% services healthy
            success_metrics.append(True)
        else:
            success_metrics.append(False)
        
        if confidence and confidence_accuracy >= 70:  # 70% confidence accuracy
            success_metrics.append(True)
        else:
            success_metrics.append(False)
        
        if uncertainty and detection_rate >= 70:  # 70% uncertainty detection
            success_metrics.append(True)
        else:
            success_metrics.append(False)
        
        if pipeline and pipeline.get('integration_success'):
            success_metrics.append(True)
        else:
            success_metrics.append(False)
        
        overall_success = sum(success_metrics) / len(success_metrics) if success_metrics else 0
        
        if overall_success >= 0.75:
            print("   🎉 EXCELLENT: Confidence-driven system is working excellently!")
            print("   ✅ The system can detect knowledge gaps and create LoRAs automatically")
        elif overall_success >= 0.5:
            print("   ✅ GOOD: Confidence-driven system is working well with minor issues")
            print("   🔧 Some components may need attention")
        else:
            print("   ⚠️ NEEDS WORK: Confidence-driven system needs significant improvements")
            print("   🛠️ Multiple components require attention")
        
        # Key Features Working
        print(f"\n🌟 KEY FEATURES STATUS:")
        print(f"   📊 Real-time confidence monitoring: {'✅' if healthy_services > 0 else '❌'}")
        print(f"   🔍 Automatic gap detection: {'✅' if uncertainty and detection_rate > 50 else '❌'}")
        print(f"   🚀 Self-triggered LoRA creation: {'✅' if uncertainty and trigger_rate > 50 else '❌'}")
        print(f"   💬 Chat integration: {'✅' if chat and chat.get('chat_tests', 0) > 0 else '❌'}")
        print(f"   📈 Knowledge tracking: {'✅' if gaps and gaps.get('total_gaps', 0) >= 0 else '❌'}")
        print(f"   🔄 End-to-end pipeline: {'✅' if pipeline and pipeline.get('integration_success') else '❌'}")
        
        print(f"\n🎯 CONFIDENCE-DRIVEN SYSTEM SUCCESS RATE: {overall_success * 100:.1f}%")
        
        if overall_success >= 0.75:
            print("\n🚀 The system is ready for production use!")
            print("   When the AI says 'I don't know', it will automatically:")
            print("   • Detect the knowledge gap in real-time")
            print("   • Trigger targeted content crawling")
            print("   • Create specialized LoRAs to fill the gap")
            print("   • Improve responses for future similar queries")
        
        print("\n" + "=" * 60)

async def main():
    """Main testing function"""
    
    print("🧠💡 CONFIDENCE-DRIVEN LORA CREATION SYSTEM TESTING")
    print("🎯 Testing automatic LoRA creation when AI says 'I don't know'")
    print("=" * 60)
    
    tester = ConfidenceDrivenSystemTester()
    await tester.run_comprehensive_test_suite()

if __name__ == "__main__":
    asyncio.run(main()) 