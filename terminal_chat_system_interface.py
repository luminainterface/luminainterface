#!/usr/bin/env python3
"""
TERMINAL CHAT SYSTEM INTERFACE - CONFIDENCE-DRIVEN LORA SYSTEM
============================================================
Comprehensive terminal chat interface for testing and validating the 
confidence-driven LoRA system with real-time metrics and HiRa integration.
"""

import asyncio
import aiohttp
import json
import time
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import colorama
from colorama import Fore, Back, Style
import threading
import queue

# Initialize colorama for Windows compatibility
colorama.init()

@dataclass
class SystemMetrics:
    """Real-time system metrics"""
    confidence_scores: List[float]
    response_times: List[float]
    knowledge_gaps_detected: int
    lora_requests_triggered: int
    services_healthy: int
    total_services: int
    average_confidence: float
    system_utilization: float
    learning_events: int
    
class TerminalChatInterface:
    """Terminal chat interface for the confidence-driven LoRA system"""
    
    def __init__(self):
        # Service endpoints
        self.endpoints = {
            'high_rank_adapter': 'http://localhost:9000',  # HiRa steering
            'ultimate_chat': 'http://localhost:8950',      # Chat with confidence
            'confidence_creator': 'http://localhost:8848',  # LoRA creator
            'neural_engine': 'http://localhost:8890',       # Neural thought engine
            'meta_orchestration': 'http://localhost:8999',  # Meta orchestration
            'enhanced_execution': 'http://localhost:8998',  # Enhanced execution
            'rag_coordination': 'http://localhost:8952',    # RAG coordination
            'v7_logic': 'http://localhost:8991',             # V7 logic agent
            'quantum_agent': 'http://localhost:8975',       # Quantum agent
            'llm_gap_detector': 'http://localhost:8997',    # LLM gap detection
            'research_agent': 'http://localhost:8999',      # Research agent V3
            'mathematical': 'http://localhost:8990'         # V5 math orchestrator
        }
        
        # Metrics tracking
        self.metrics = SystemMetrics(
            confidence_scores=[],
            response_times=[],
            knowledge_gaps_detected=0,
            lora_requests_triggered=0,
            services_healthy=0,
            total_services=len(self.endpoints),
            average_confidence=0.0,
            system_utilization=0.0,
            learning_events=0
        )
        
        # Chat history and session
        self.chat_history = []
        self.session_id = f"terminal_session_{int(time.time())}"
        self.conversation_count = 0
        
        # System monitoring
        self.monitoring_active = True
        self.metrics_queue = queue.Queue()
        
        # HiRa steering parameters
        self.hira_config = {
            'transcript_influence': 0.8,
            'pattern_sensitivity': 0.7,
            'evolution_aggressiveness': 0.6,
            'self_reflection_depth': 0.9,
            'quality_prioritization': 0.85,
            'confidence_steering_enabled': True,
            'learning_acceleration': True
        }
    
    async def initialize_system(self):
        """Initialize and health check all system components"""
        print(f"{Fore.CYAN}🧠💡 CONFIDENCE-DRIVEN LORA SYSTEM - TERMINAL INTERFACE{Style.RESET_ALL}")
        print("=" * 70)
        print(f"{Fore.YELLOW}Initializing system connections...{Style.RESET_ALL}")
        
        # Health check all services
        healthy_services = 0
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            for service_name, endpoint in self.endpoints.items():
                try:
                    async with session.get(f"{endpoint}/health") as response:
                        if response.status == 200:
                            print(f"  ✅ {service_name}: {endpoint}")
                            healthy_services += 1
                        else:
                            print(f"  ⚠️  {service_name}: {endpoint} (Status: {response.status})")
                except Exception as e:
                    print(f"  ❌ {service_name}: {endpoint} (Error: {str(e)[:50]})")
        
        self.metrics.services_healthy = healthy_services
        self.metrics.system_utilization = (healthy_services / self.metrics.total_services) * 100
        
        print(f"\n{Fore.GREEN}System Status: {healthy_services}/{self.metrics.total_services} services healthy ({self.metrics.system_utilization:.1f}%){Style.RESET_ALL}")
        
        # Initialize HiRa steering
        await self.initialize_hira_steering()
        
        # Start background monitoring
        self.start_background_monitoring()
        
        print(f"{Fore.CYAN}System ready for testing! 🚀{Style.RESET_ALL}\n")
    
    async def initialize_hira_steering(self):
        """Initialize High-Rank Adapter for system steering"""
        try:
            async with aiohttp.ClientSession() as session:
                # Configure HiRa for confidence-driven steering
                config_payload = {
                    'session_id': self.session_id,
                    'steering_config': self.hira_config,
                    'confidence_integration': True,
                    'learning_feedback_enabled': True
                }
                
                async with session.post(
                    f"{self.endpoints['high_rank_adapter']}/configure", 
                    json=config_payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        print(f"  ✅ HiRa Steering: Configured with confidence integration")
                        print(f"     📊 Steering Sensitivity: {self.hira_config['pattern_sensitivity']}")
                        print(f"     🧠 Learning Acceleration: {self.hira_config['learning_acceleration']}")
                    else:
                        print(f"  ⚠️  HiRa Steering: Configuration warning (Status: {response.status})")
                        
        except Exception as e:
            print(f"  ⚠️  HiRa Steering: {str(e)[:60]}")
    
    def start_background_monitoring(self):
        """Start background monitoring of system metrics"""
        def monitor_metrics():
            while self.monitoring_active:
                try:
                    # This would run in background to collect metrics
                    asyncio.run(self.collect_metrics())
                    time.sleep(10)  # Collect metrics every 10 seconds
                except Exception as e:
                    pass  # Silent failure for background monitoring
        
        monitoring_thread = threading.Thread(target=monitor_metrics, daemon=True)
        monitoring_thread.start()
    
    async def collect_metrics(self):
        """Collect real-time metrics from system services"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as session:
                # Get confidence analytics
                try:
                    async with session.get(f"{self.endpoints['confidence_creator']}/confidence_analytics") as response:
                        if response.status == 200:
                            analytics = await response.json()
                            if 'average_confidence' in analytics:
                                self.metrics.average_confidence = analytics['average_confidence']
                            if 'knowledge_gaps_detected' in analytics:
                                self.metrics.knowledge_gaps_detected = analytics['knowledge_gaps_detected']
                            if 'lora_requests_triggered' in analytics:
                                self.metrics.lora_requests_triggered = analytics['lora_requests_triggered']
                except:
                    pass
                
                # Get knowledge gaps
                try:
                    async with session.get(f"{self.endpoints['confidence_creator']}/knowledge_gaps") as response:
                        if response.status == 200:
                            gaps = await response.json()
                            if isinstance(gaps, list):
                                self.metrics.knowledge_gaps_detected = len(gaps)
                except:
                    pass
                    
        except Exception as e:
            pass  # Silent failure for metrics collection
    
    def display_metrics_dashboard(self):
        """Display real-time metrics dashboard"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print(f"{Fore.CYAN}🧠💡 CONFIDENCE-DRIVEN LORA SYSTEM - LIVE METRICS{Style.RESET_ALL}")
        print("=" * 70)
        
        # System health
        health_color = Fore.GREEN if self.metrics.system_utilization > 80 else Fore.YELLOW if self.metrics.system_utilization > 50 else Fore.RED
        print(f"{health_color}🏥 System Health: {self.metrics.services_healthy}/{self.metrics.total_services} services ({self.metrics.system_utilization:.1f}%){Style.RESET_ALL}")
        
        # Confidence metrics
        conf_color = Fore.GREEN if self.metrics.average_confidence > 0.7 else Fore.YELLOW if self.metrics.average_confidence > 0.4 else Fore.RED
        print(f"{conf_color}📊 Average Confidence: {self.metrics.average_confidence:.3f}{Style.RESET_ALL}")
        
        # Learning metrics
        print(f"{Fore.BLUE}🎯 Knowledge Gaps Detected: {self.metrics.knowledge_gaps_detected}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}🔄 LoRA Requests Triggered: {self.metrics.lora_requests_triggered}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}📚 Learning Events: {self.metrics.learning_events}{Style.RESET_ALL}")
        
        # Performance metrics
        if self.metrics.response_times:
            avg_response = sum(self.metrics.response_times) / len(self.metrics.response_times)
            response_color = Fore.GREEN if avg_response < 2.0 else Fore.YELLOW if avg_response < 5.0 else Fore.RED
            print(f"{response_color}⚡ Average Response Time: {avg_response:.2f}s{Style.RESET_ALL}")
        
        # Recent confidence scores
        if self.metrics.confidence_scores:
            recent_scores = self.metrics.confidence_scores[-5:]
            print(f"{Fore.MAGENTA}📈 Recent Confidence Scores: {[f'{s:.2f}' for s in recent_scores]}{Style.RESET_ALL}")
        
        # Session info
        print(f"{Fore.CYAN}🗣️  Conversation Count: {self.conversation_count}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}🕒 Session ID: {self.session_id}{Style.RESET_ALL}")
        print("=" * 70)
    
    async def send_message_with_hira_steering(self, message: str) -> Dict[str, Any]:
        """Send message through the entire system with HiRa steering"""
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                # Step 1: HiRa preprocessing and steering
                hira_payload = {
                    'message': message,
                    'session_id': self.session_id,
                    'conversation_count': self.conversation_count,
                    'previous_confidence_scores': self.metrics.confidence_scores[-3:] if self.metrics.confidence_scores else [],
                    'steering_enabled': True
                }
                
                hira_response = None
                try:
                    async with session.post(
                        f"{self.endpoints['high_rank_adapter']}/process", 
                        json=hira_payload
                    ) as response:
                        if response.status == 200:
                            hira_response = await response.json()
                        else:
                            print(f"{Fore.YELLOW}⚠️  HiRa preprocessing unavailable, proceeding direct{Style.RESET_ALL}")
                except:
                    print(f"{Fore.YELLOW}⚠️  HiRa unavailable, using direct routing{Style.RESET_ALL}")
                
                # Step 2: Send to Ultimate Chat Orchestrator with Confidence
                chat_payload = {
                    'message': message,
                    'session_id': self.session_id,
                    'confidence_reporting': True,
                    'learning_enabled': True,
                    'hira_preprocessing': hira_response,
                    'metrics_tracking': True
                }
                
                async with session.post(
                    f"{self.endpoints['ultimate_chat']}/chat", 
                    json=chat_payload
                ) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        # Extract metrics
                        confidence = result.get('confidence_score', 0.0)
                        self.metrics.confidence_scores.append(confidence)
                        self.metrics.response_times.append(response_time)
                        
                        # Track learning events
                        if result.get('knowledge_gap_detected'):
                            self.metrics.knowledge_gaps_detected += 1
                        if result.get('lora_request_triggered'):
                            self.metrics.lora_requests_triggered += 1
                            self.metrics.learning_events += 1
                        
                        # Update average confidence
                        if self.metrics.confidence_scores:
                            self.metrics.average_confidence = sum(self.metrics.confidence_scores) / len(self.metrics.confidence_scores)
                        
                        return {
                            'response': result.get('response', 'No response received'),
                            'confidence_score': confidence,
                            'response_time': response_time,
                            'knowledge_gap_detected': result.get('knowledge_gap_detected', False),
                            'lora_request_triggered': result.get('lora_request_triggered', False),
                            'services_used': result.get('services_used', []),
                            'system_insights': result.get('system_insights', {}),
                            'hira_steering_applied': hira_response is not None
                        }
                    else:
                        error_text = await response.text()
                        return {
                            'response': f"Error: HTTP {response.status} - {error_text[:100]}",
                            'confidence_score': 0.0,
                            'response_time': response_time,
                            'error': True
                        }
                        
        except Exception as e:
            response_time = time.time() - start_time
            return {
                'response': f"Connection Error: {str(e)[:100]}",
                'confidence_score': 0.0,
                'response_time': response_time,
                'error': True
            }
    
    def display_response_with_metrics(self, message: str, result: Dict[str, Any]):
        """Display response with comprehensive metrics"""
        print(f"\n{Fore.BLUE}📤 You:{Style.RESET_ALL} {message}")
        print(f"{Fore.GREEN}🤖 System:{Style.RESET_ALL} {result['response']}")
        
        # Confidence visualization
        confidence = result.get('confidence_score', 0.0)
        conf_bar = "█" * int(confidence * 20) + "░" * (20 - int(confidence * 20))
        conf_color = Fore.GREEN if confidence > 0.7 else Fore.YELLOW if confidence > 0.4 else Fore.RED
        print(f"{conf_color}📊 Confidence: {confidence:.3f} [{conf_bar}]{Style.RESET_ALL}")
        
        # Response time
        response_time = result.get('response_time', 0.0)
        time_color = Fore.GREEN if response_time < 2.0 else Fore.YELLOW if response_time < 5.0 else Fore.RED
        print(f"{time_color}⚡ Response Time: {response_time:.2f}s{Style.RESET_ALL}")
        
        # Learning indicators
        if result.get('knowledge_gap_detected'):
            print(f"{Fore.MAGENTA}🎯 Knowledge Gap Detected - Learning Triggered!{Style.RESET_ALL}")
        
        if result.get('lora_request_triggered'):
            print(f"{Fore.CYAN}🔄 LoRA Creation Requested - System Learning...{Style.RESET_ALL}")
        
        if result.get('hira_steering_applied'):
            print(f"{Fore.BLUE}🧠 HiRa Steering Applied - Enhanced Processing{Style.RESET_ALL}")
        
        # Services used
        services_used = result.get('services_used', [])
        if services_used:
            print(f"{Fore.YELLOW}🔧 Services: {', '.join(services_used)}{Style.RESET_ALL}")
        
        # System insights
        insights = result.get('system_insights', {})
        if insights:
            print(f"{Fore.CYAN}💡 Insights: {json.dumps(insights, indent=2)}{Style.RESET_ALL}")
        
        print("-" * 70)
    
    async def run_system_diagnostics(self):
        """Run comprehensive system diagnostics"""
        print(f"\n{Fore.CYAN}🔧 RUNNING SYSTEM DIAGNOSTICS...{Style.RESET_ALL}")
        print("=" * 50)
        
        # Test confidence detection
        test_queries = [
            ("What is 2 + 2?", "High confidence expected"),
            ("I don't know about quantum computing", "Should trigger gap detection"),
            ("Explain the latest developments in AI from yesterday", "Low confidence expected"),
            ("What is the ZetaML algorithm?", "Should trigger LoRA creation"),
        ]
        
        for query, expectation in test_queries:
            print(f"\n{Fore.YELLOW}Testing: {query}{Style.RESET_ALL}")
            print(f"Expected: {expectation}")
            
            result = await self.send_message_with_hira_steering(query)
            confidence = result.get('confidence_score', 0.0)
            
            if "don't know" in query.lower() or "latest" in query.lower() or "ZetaML" in query:
                if confidence < 0.5 or result.get('knowledge_gap_detected'):
                    print(f"  ✅ Gap detection working correctly (confidence: {confidence:.3f})")
                else:
                    print(f"  ⚠️  Gap detection may need tuning (confidence: {confidence:.3f})")
            else:
                if confidence > 0.7:
                    print(f"  ✅ High confidence response working (confidence: {confidence:.3f})")
                else:
                    print(f"  ⚠️  Expected higher confidence (confidence: {confidence:.3f})")
        
        print(f"\n{Fore.GREEN}Diagnostics completed!{Style.RESET_ALL}")
    
    async def interactive_chat_session(self):
        """Run interactive chat session with metrics"""
        print(f"\n{Fore.GREEN}🗣️  INTERACTIVE CHAT SESSION STARTED{Style.RESET_ALL}")
        print("Commands: /metrics - show dashboard, /diagnostics - run tests, /quit - exit")
        print("=" * 70)
        
        while True:
            try:
                # Get user input
                user_input = input(f"\n{Fore.BLUE}You: {Style.RESET_ALL}").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() == '/quit':
                    break
                elif user_input.lower() == '/metrics':
                    self.display_metrics_dashboard()
                    input(f"\n{Fore.CYAN}Press Enter to continue...{Style.RESET_ALL}")
                    continue
                elif user_input.lower() == '/diagnostics':
                    await self.run_system_diagnostics()
                    continue
                
                # Send message through system
                self.conversation_count += 1
                result = await self.send_message_with_hira_steering(user_input)
                
                # Display response with metrics
                self.display_response_with_metrics(user_input, result)
                
                # Add to chat history
                self.chat_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'user_message': user_input,
                    'system_response': result['response'],
                    'confidence_score': result.get('confidence_score', 0.0),
                    'response_time': result.get('response_time', 0.0),
                    'knowledge_gap_detected': result.get('knowledge_gap_detected', False),
                    'lora_request_triggered': result.get('lora_request_triggered', False)
                })
                
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}Chat session interrupted by user{Style.RESET_ALL}")
                break
            except Exception as e:
                print(f"\n{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
    
    async def save_session_report(self):
        """Save comprehensive session report"""
        report = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'metrics': asdict(self.metrics),
            'hira_config': self.hira_config,
            'chat_history': self.chat_history,
            'conversation_count': self.conversation_count,
            'system_performance': {
                'avg_confidence': self.metrics.average_confidence,
                'avg_response_time': sum(self.metrics.response_times) / len(self.metrics.response_times) if self.metrics.response_times else 0.0,
                'learning_efficiency': self.metrics.learning_events / max(1, self.conversation_count),
                'system_utilization': self.metrics.system_utilization
            }
        }
        
        report_file = f"session_report_{self.session_id}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n{Fore.GREEN}📄 Session report saved: {report_file}{Style.RESET_ALL}")
        return report_file
    
    def cleanup(self):
        """Cleanup and shutdown"""
        self.monitoring_active = False
        print(f"\n{Fore.CYAN}🧠💡 Confidence-Driven System Interface Shutdown{Style.RESET_ALL}")
        print(f"Final Metrics:")
        print(f"  Conversations: {self.conversation_count}")
        print(f"  Average Confidence: {self.metrics.average_confidence:.3f}")
        print(f"  Knowledge Gaps Detected: {self.metrics.knowledge_gaps_detected}")
        print(f"  LoRA Requests: {self.metrics.lora_requests_triggered}")
        print(f"  System Utilization: {self.metrics.system_utilization:.1f}%")

async def main():
    """Main entry point"""
    interface = TerminalChatInterface()
    
    try:
        # Initialize system
        await interface.initialize_system()
        
        # Run interactive session
        await interface.interactive_chat_session()
        
        # Save session report
        await interface.save_session_report()
        
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}System shutdown requested{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}System error: {str(e)}{Style.RESET_ALL}")
    finally:
        interface.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 