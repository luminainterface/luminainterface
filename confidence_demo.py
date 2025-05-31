#!/usr/bin/env python3
"""
CONFIDENCE-DRIVEN LORA CREATION DEMO
===================================
Standalone demonstration of how the confidence system works:
- Detects when AI doesn't know something
- Automatically triggers LoRA creation for knowledge gaps
- Shows real-time confidence monitoring
"""

import time
import json
import random
from datetime import datetime
from typing import Dict, Any, List

class ConfidenceDrivenDemo:
    """Demonstration of confidence-driven LoRA creation"""
    
    def __init__(self):
        self.knowledge_gaps = []
        self.lora_creation_requests = []
        self.confidence_history = []
        
        # Thresholds for triggering LoRA creation
        self.lora_creation_threshold = 0.3
        self.urgent_lora_threshold = 0.1
        
        # Demo queries with expected confidence levels
        self.demo_queries = [
            {
                'query': "What is 2 + 2?",
                'expected_confidence': 0.95,
                'should_trigger_lora': False,
                'expected_response': "The answer is 4."
            },
            {
                'query': "Explain machine learning basics",
                'expected_confidence': 0.85,
                'should_trigger_lora': False,
                'expected_response': "Machine learning is a subset of AI that enables computers to learn..."
            },
            {
                'query': "What are the latest developments in quantum computing this week?",
                'expected_confidence': 0.25,
                'should_trigger_lora': True,
                'expected_response': "I don't have access to the most recent quantum computing developments from this week."
            },
            {
                'query': "How does the fictional ZetaML algorithm work?",
                'expected_confidence': 0.05,
                'should_trigger_lora': True,
                'expected_response': "I don't have information about the ZetaML algorithm as it may be fictional or very specialized."
            },
            {
                'query': "Can you explain quantum-biological computing interfaces?",
                'expected_confidence': 0.15,
                'should_trigger_lora': True,
                'expected_response': "I don't have enough detailed information about quantum-biological computing interfaces."
            }
        ]
    
    def run_demo(self):
        """Run the complete confidence-driven demo"""
        
        print("🧠💡 CONFIDENCE-DRIVEN LORA CREATION SYSTEM DEMO")
        print("🎯 Demonstrating automatic LoRA creation when AI says 'I don't know'")
        print("=" * 70)
        
        print("\n📋 CONFIDENCE THRESHOLDS:")
        print(f"   LoRA Creation Threshold: {self.lora_creation_threshold}")
        print(f"   Urgent LoRA Threshold: {self.urgent_lora_threshold}")
        print(f"   High Confidence: 0.8+")
        print(f"   Medium Confidence: 0.4-0.8")
        print(f"   Low Confidence: 0.2-0.4")
        print(f"   Very Low/Unknown: < 0.2")
        
        print("\n🔄 RUNNING CONFIDENCE ASSESSMENT DEMO...")
        print("=" * 50)
        
        for i, demo_case in enumerate(self.demo_queries, 1):
            print(f"\n{i}. 💬 Query: \"{demo_case['query']}\"")
            
            # Simulate confidence assessment
            confidence_result = self._assess_confidence(
                demo_case['query'],
                demo_case['expected_response'],
                demo_case['expected_confidence']
            )
            
            # Display results
            self._display_confidence_result(confidence_result)
            
            # Check if LoRA should be triggered
            if confidence_result['should_trigger_lora']:
                lora_request = self._trigger_lora_creation(demo_case['query'], confidence_result)
                self._display_lora_request(lora_request)
            
            # Add delay for demonstration effect
            time.sleep(1)
        
        # Show final summary
        self._show_demo_summary()
    
    def _assess_confidence(self, query: str, response: str, expected_confidence: float) -> Dict[str, Any]:
        """Simulate confidence assessment"""
        
        # Simulate confidence calculation with some variation
        confidence_score = expected_confidence + random.uniform(-0.05, 0.05)
        confidence_score = max(0.0, min(1.0, confidence_score))
        
        # Determine confidence level
        if confidence_score >= 0.8:
            confidence_level = "high"
        elif confidence_score >= 0.4:
            confidence_level = "medium"
        elif confidence_score >= 0.2:
            confidence_level = "low"
        else:
            confidence_level = "very_low"
        
        # Check if LoRA should be triggered
        should_trigger_lora = confidence_score <= self.lora_creation_threshold
        is_urgent = confidence_score <= self.urgent_lora_threshold
        
        # Detect uncertainty keywords
        uncertainty_keywords = [
            "don't know", "don't have", "unclear", "uncertain", 
            "not sure", "can't determine", "insufficient information",
            "fictional", "may be", "not familiar"
        ]
        has_uncertainty_keywords = any(keyword in response.lower() for keyword in uncertainty_keywords)
        
        # Assess gap characteristics
        gap_characteristics = self._analyze_knowledge_gap(query, response, confidence_score)
        
        result = {
            'query': query,
            'response': response,
            'confidence_score': round(confidence_score, 3),
            'confidence_level': confidence_level,
            'should_trigger_lora': should_trigger_lora,
            'is_urgent': is_urgent,
            'has_uncertainty_keywords': has_uncertainty_keywords,
            'gap_characteristics': gap_characteristics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store in history
        self.confidence_history.append(result)
        
        return result
    
    def _analyze_knowledge_gap(self, query: str, response: str, confidence_score: float) -> Dict[str, Any]:
        """Analyze the characteristics of a knowledge gap"""
        
        # Determine domain
        domain = "general"
        if any(term in query.lower() for term in ["quantum", "physics"]):
            domain = "physics"
        elif any(term in query.lower() for term in ["machine learning", "ai", "algorithm"]):
            domain = "ai_ml"
        elif any(term in query.lower() for term in ["biology", "bio", "computing"]):
            domain = "biology"
        elif any(term in query.lower() for term in ["math", "calculate", "+"]):
            domain = "mathematics"
        
        # Assess urgency
        urgency = "low"
        if confidence_score <= 0.1:
            urgency = "critical"
        elif confidence_score <= 0.2:
            urgency = "high"
        elif confidence_score <= 0.3:
            urgency = "medium"
        
        # Determine gap type
        gap_type = "knowledge"
        if "fictional" in response.lower() or "made-up" in response.lower():
            gap_type = "non_existent"
        elif "latest" in query.lower() or "recent" in query.lower():
            gap_type = "temporal"
        elif "how does" in query.lower() or "explain" in query.lower():
            gap_type = "understanding"
        
        return {
            'domain': domain,
            'urgency': urgency,
            'gap_type': gap_type,
            'complexity': "high" if len(query.split()) > 8 else "medium"
        }
    
    def _trigger_lora_creation(self, query: str, confidence_result: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate LoRA creation triggering"""
        
        gap_id = f"gap_{len(self.knowledge_gaps) + 1}_{int(time.time())}"
        
        # Create knowledge gap record
        knowledge_gap = {
            'gap_id': gap_id,
            'query': query,
            'confidence_score': confidence_result['confidence_score'],
            'domain': confidence_result['gap_characteristics']['domain'],
            'urgency': confidence_result['gap_characteristics']['urgency'],
            'gap_type': confidence_result['gap_characteristics']['gap_type'],
            'detected_at': datetime.now().isoformat(),
            'status': 'pending_lora_creation'
        }
        
        self.knowledge_gaps.append(knowledge_gap)
        
        # Create LoRA request
        lora_request = {
            'request_id': f"lora_req_{len(self.lora_creation_requests) + 1}",
            'gap_id': gap_id,
            'target_domain': confidence_result['gap_characteristics']['domain'],
            'urgency': confidence_result['gap_characteristics']['urgency'],
            'search_keywords': self._extract_keywords(query),
            'expected_lora_size': 'small' if confidence_result['gap_characteristics']['complexity'] == 'medium' else 'large',
            'priority_score': self._calculate_priority_score(confidence_result),
            'status': 'initiated',
            'created_at': datetime.now().isoformat()
        }
        
        self.lora_creation_requests.append(lora_request)
        
        return lora_request
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract relevant keywords for LoRA training"""
        
        # Simple keyword extraction
        stop_words = {'what', 'is', 'how', 'does', 'the', 'a', 'an', 'and', 'or', 'but', 'can', 'you'}
        words = query.lower().replace('?', '').split()
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords[:5]  # Top 5 keywords
    
    def _calculate_priority_score(self, confidence_result: Dict[str, Any]) -> float:
        """Calculate priority score for LoRA creation"""
        
        base_priority = 1.0 - confidence_result['confidence_score']  # Lower confidence = higher priority
        
        # Urgency multiplier
        urgency_multipliers = {
            'critical': 2.0,
            'high': 1.5,
            'medium': 1.2,
            'low': 1.0
        }
        
        urgency = confidence_result['gap_characteristics']['urgency']
        priority_score = base_priority * urgency_multipliers.get(urgency, 1.0)
        
        # Boost for uncertainty keywords
        if confidence_result['has_uncertainty_keywords']:
            priority_score *= 1.3
        
        return round(min(priority_score, 2.0), 3)  # Cap at 2.0
    
    def _display_confidence_result(self, result: Dict[str, Any]):
        """Display confidence assessment results"""
        
        confidence_score = result['confidence_score']
        confidence_level = result['confidence_level']
        
        # Color coding for display
        if confidence_level == "high":
            level_emoji = "🟢"
        elif confidence_level == "medium":
            level_emoji = "🟡"
        elif confidence_level == "low":
            level_emoji = "🟠"
        else:
            level_emoji = "🔴"
        
        print(f"   📊 Confidence: {level_emoji} {confidence_level.upper()} ({confidence_score:.3f})")
        print(f"   💭 Response: \"{result['response']}\"")
        
        if result['has_uncertainty_keywords']:
            print(f"   🔍 Uncertainty detected: Keywords found")
        
        gap_chars = result['gap_characteristics']
        print(f"   🎯 Domain: {gap_chars['domain']} | Urgency: {gap_chars['urgency']} | Type: {gap_chars['gap_type']}")
        
        if result['should_trigger_lora']:
            urgency_text = "🚨 URGENT" if result['is_urgent'] else "⚡ TRIGGERED"
            print(f"   🚀 LoRA Creation: {urgency_text}")
        else:
            print(f"   ✅ LoRA Creation: Not needed (confidence sufficient)")
    
    def _display_lora_request(self, lora_request: Dict[str, Any]):
        """Display LoRA creation request details"""
        
        print(f"   📝 LoRA Request Details:")
        print(f"      ID: {lora_request['request_id']}")
        print(f"      Domain: {lora_request['target_domain']}")
        print(f"      Priority: {lora_request['priority_score']}")
        print(f"      Keywords: {', '.join(lora_request['search_keywords'])}")
        print(f"      Size: {lora_request['expected_lora_size']}")
        print(f"      Status: {lora_request['status']}")
    
    def _show_demo_summary(self):
        """Show demonstration summary"""
        
        print("\n" + "=" * 70)
        print("🏆 CONFIDENCE-DRIVEN DEMO SUMMARY")
        print("=" * 70)
        
        total_queries = len(self.demo_queries)
        loras_triggered = len(self.lora_creation_requests)
        knowledge_gaps_detected = len(self.knowledge_gaps)
        
        print(f"\n📊 DEMO STATISTICS:")
        print(f"   Total queries processed: {total_queries}")
        print(f"   Knowledge gaps detected: {knowledge_gaps_detected}")
        print(f"   LoRAs triggered: {loras_triggered}")
        print(f"   LoRA trigger rate: {(loras_triggered/total_queries)*100:.1f}%")
        
        # Confidence distribution
        high_conf = sum(1 for h in self.confidence_history if h['confidence_level'] == 'high')
        medium_conf = sum(1 for h in self.confidence_history if h['confidence_level'] == 'medium')
        low_conf = sum(1 for h in self.confidence_history if h['confidence_level'] == 'low')
        very_low_conf = sum(1 for h in self.confidence_history if h['confidence_level'] == 'very_low')
        
        print(f"\n🎯 CONFIDENCE DISTRIBUTION:")
        print(f"   🟢 High confidence: {high_conf} queries")
        print(f"   🟡 Medium confidence: {medium_conf} queries")
        print(f"   🟠 Low confidence: {low_conf} queries")
        print(f"   🔴 Very low confidence: {very_low_conf} queries")
        
        # Knowledge gap domains
        if self.knowledge_gaps:
            domains = [gap['domain'] for gap in self.knowledge_gaps]
            domain_counts = {domain: domains.count(domain) for domain in set(domains)}
            
            print(f"\n🎯 KNOWLEDGE GAP DOMAINS:")
            for domain, count in domain_counts.items():
                print(f"   📚 {domain}: {count} gaps")
        
        # LoRA requests by urgency
        if self.lora_creation_requests:
            urgencies = [req['urgency'] for req in self.lora_creation_requests]
            urgency_counts = {urgency: urgencies.count(urgency) for urgency in set(urgencies)}
            
            print(f"\n🚀 LORA REQUESTS BY URGENCY:")
            for urgency, count in urgency_counts.items():
                emoji = {"critical": "🚨", "high": "⚡", "medium": "🔔", "low": "📝"}.get(urgency, "📝")
                print(f"   {emoji} {urgency}: {count} requests")
        
        print(f"\n🌟 KEY CONFIDENCE-DRIVEN FEATURES DEMONSTRATED:")
        print(f"   ✅ Real-time confidence monitoring")
        print(f"   ✅ Automatic knowledge gap detection")
        print(f"   ✅ Self-triggered LoRA creation when AI says 'I don't know'")
        print(f"   ✅ Intelligent uncertainty pattern recognition")
        print(f"   ✅ Domain-specific gap analysis")
        print(f"   ✅ Priority-based LoRA scheduling")
        print(f"   ✅ Demand-driven learning pipeline")
        
        print(f"\n🎯 CONFIDENCE SYSTEM EFFECTIVENESS:")
        if loras_triggered > 0:
            print(f"   🎉 EXCELLENT: System successfully detected {knowledge_gaps_detected} knowledge gaps!")
            print(f"   ✅ Automatically triggered {loras_triggered} LoRA creation requests")
            print(f"   🚀 The system is working as designed - creating LoRAs only when needed")
        else:
            print(f"   ℹ️  No LoRAs triggered - all queries had sufficient confidence")
        
        print(f"\n💡 HOW IT WORKS IN PRODUCTION:")
        print(f"   1. 🔍 Every AI response is monitored for confidence level")
        print(f"   2. 🎯 When confidence drops below {self.lora_creation_threshold}, gap is detected")
        print(f"   3. 🚀 LoRA creation is automatically triggered for that specific knowledge area")
        print(f"   4. 📚 Enhanced crawler finds relevant content for training")
        print(f"   5. 🔧 Concept training worker creates specialized LoRA")
        print(f"   6. ✅ Future similar queries get improved responses")
        
        print(f"\n🔄 THIS IS MUCH BETTER THAN TIMER-BASED LORA CREATION BECAUSE:")
        print(f"   ✅ Demand-driven: Only creates LoRAs when actually needed")
        print(f"   ✅ Targeted: Focuses on specific knowledge gaps, not random topics")
        print(f"   ✅ Efficient: No wasted resources on unnecessary training")
        print(f"   ✅ Responsive: Immediately addresses detected weaknesses")
        print(f"   ✅ Intelligent: Uses AI uncertainty as a learning signal")
        
        print("\n" + "=" * 70)

def main():
    """Run the confidence-driven demo"""
    
    demo = ConfidenceDrivenDemo()
    demo.run_demo()

if __name__ == "__main__":
    main() 