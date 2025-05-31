#!/usr/bin/env python3
"""
ULTIMATE CHAT ORCHESTRATOR WITH CONFIDENCE INTEGRATION
====================================================
Enhanced chat orchestrator that:
- Monitors its own confidence levels in real-time
- Detects when it doesn't know something
- Automatically triggers LoRA creation for knowledge gaps
- Learns and improves from uncertainty patterns
- Provides transparent confidence reporting to users
"""

import os
import time
import json
import uuid
import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import re

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PORT = int(os.getenv('PORT', '8950'))

# Initialize FastAPI
app = FastAPI(
    title="Ultimate Chat Orchestrator with Confidence",
    description="AI chat system with automatic knowledge gap detection and LoRA creation",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    confidence_reporting: bool = True

class ChatResponse(BaseModel):
    response: str
    confidence_score: float
    confidence_level: str
    knowledge_gap_detected: bool
    lora_triggered: bool
    recommendations: List[str]
    conversation_id: str
    timestamp: str

# Global state for the orchestrator
orchestrator_state = {
    'conversations': {},           # conversation_id -> conversation history
    'user_sessions': {},          # user_id -> session data
    'confidence_patterns': {},    # patterns of low confidence
    'learning_requests': {},      # active learning/LoRA requests
    'orchestrator_stats': {
        'total_messages': 0,
        'low_confidence_responses': 0,
        'loras_triggered': 0,
        'knowledge_gaps_detected': 0,
        'avg_confidence': 0.0
    }
}

class UltimateChatOrchestratorWithConfidence:
    """Enhanced chat orchestrator with confidence monitoring"""
    
    def __init__(self):
        self.logger = logging.getLogger("UltimateChatOrchestrator")
        
        # Service endpoints
        self.service_endpoints = {
            'confidence_lora_creator': 'http://localhost:8848',
            'llm_gap_detector': 'http://localhost:8997',
            'neural_engine': 'http://localhost:8890',
            'rag_coordination': 'http://localhost:8952',
            'lora_coordination': 'http://localhost:8995',
            'enhanced_fact_checker': 'http://localhost:8885',
            'v7_logic_agent': 'http://localhost:8991',
            'quantum_agent': 'http://localhost:8975'
        }
        
        # Confidence thresholds
        self.confidence_config = {
            'low_confidence_threshold': 0.4,
            'very_low_confidence_threshold': 0.2,
            'uncertainty_trigger_threshold': 0.3,
            'enable_automatic_lora': True,
            'enable_transparency': True,
            'max_learning_attempts_per_topic': 3
        }
        
        # Uncertainty detection patterns
        self.uncertainty_indicators = [
            r"i don't know",
            r"i'm not sure",
            r"i don't have information",
            r"i cannot provide",
            r"i'm not familiar",
            r"i don't have access",
            r"i'm unable to",
            r"i lack knowledge",
            r"i'm uncertain",
            r"i cannot determine",
            r"i'm not aware",
            r"i don't understand",
            r"that's beyond my knowledge",
            r"i need more information",
            r"i can't help with that",
            r"i don't have data on that"
        ]
        
        # Response confidence analyzers
        self.low_confidence_patterns = [
            r"probably",
            r"possibly",
            r"might be",
            r"could be",
            r"perhaps",
            r"seems like",
            r"appears to",
            r"i think",
            r"i believe",
            r"it's likely",
            r"presumably",
            r"supposedly"
        ]
    
    async def process_chat_message(self, request: ChatRequest) -> ChatResponse:
        """Process a chat message with confidence monitoring"""
        
        start_time = time.time()
        
        # Update statistics
        orchestrator_state['orchestrator_stats']['total_messages'] += 1
        
        # Generate or retrieve conversation ID
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # Initialize conversation if new
        if conversation_id not in orchestrator_state['conversations']:
            orchestrator_state['conversations'][conversation_id] = {
                'messages': [],
                'user_id': request.user_id,
                'started_at': datetime.now().isoformat(),
                'confidence_history': []
            }
        
        conversation = orchestrator_state['conversations'][conversation_id]
        
        # Add user message to conversation
        conversation['messages'].append({
            'role': 'user',
            'content': request.message,
            'timestamp': datetime.now().isoformat()
        })
        
        try:
            # Generate AI response using multiple agents
            ai_response, confidence_score = await self._generate_ai_response(
                request.message, conversation, request.context
            )
            
            # Analyze confidence level
            confidence_level = self._determine_confidence_level(confidence_score, ai_response)
            
            # Detect knowledge gaps and uncertainty
            gap_detected, gap_info = await self._analyze_for_knowledge_gaps(
                request.message, ai_response, confidence_score
            )
            
            # Trigger LoRA creation if needed
            lora_triggered = False
            if gap_detected and self.confidence_config['enable_automatic_lora']:
                lora_triggered = await self._trigger_learning_for_gap(gap_info, conversation_id)
            
            # Generate recommendations
            recommendations = self._generate_user_recommendations(
                confidence_score, gap_detected, ai_response
            )
            
            # Add AI response to conversation
            conversation['messages'].append({
                'role': 'assistant',
                'content': ai_response,
                'confidence_score': confidence_score,
                'confidence_level': confidence_level,
                'gap_detected': gap_detected,
                'lora_triggered': lora_triggered,
                'timestamp': datetime.now().isoformat()
            })
            
            # Update confidence history
            conversation['confidence_history'].append({
                'message_pair': len(conversation['messages']) // 2,
                'confidence_score': confidence_score,
                'gap_detected': gap_detected,
                'timestamp': datetime.now().isoformat()
            })
            
            # Update statistics
            if confidence_score < self.confidence_config['low_confidence_threshold']:
                orchestrator_state['orchestrator_stats']['low_confidence_responses'] += 1
            
            if gap_detected:
                orchestrator_state['orchestrator_stats']['knowledge_gaps_detected'] += 1
            
            if lora_triggered:
                orchestrator_state['orchestrator_stats']['loras_triggered'] += 1
            
            # Update average confidence
            total_msgs = orchestrator_state['orchestrator_stats']['total_messages']
            current_avg = orchestrator_state['orchestrator_stats']['avg_confidence']
            new_avg = ((current_avg * (total_msgs - 1)) + confidence_score) / total_msgs
            orchestrator_state['orchestrator_stats']['avg_confidence'] = new_avg
            
            # Send confidence assessment to confidence creator
            if self.confidence_config['enable_automatic_lora']:
                asyncio.create_task(self._report_confidence_assessment(
                    request.message, ai_response, confidence_score, 
                    time.time() - start_time, conversation_id
                ))
            
            response_time = time.time() - start_time
            
            return ChatResponse(
                response=ai_response,
                confidence_score=confidence_score,
                confidence_level=confidence_level,
                knowledge_gap_detected=gap_detected,
                lora_triggered=lora_triggered,
                recommendations=recommendations,
                conversation_id=conversation_id,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"Error processing chat message: {e}")
            
            # Return error response with low confidence
            error_response = f"I encountered an error processing your request: {str(e)}"
            
            return ChatResponse(
                response=error_response,
                confidence_score=0.0,
                confidence_level="error",
                knowledge_gap_detected=True,
                lora_triggered=False,
                recommendations=["Please try rephrasing your question", "Check back later for improved responses"],
                conversation_id=conversation_id,
                timestamp=datetime.now().isoformat()
            )
    
    async def _generate_ai_response(self, user_message: str, conversation: Dict[str, Any], 
                                  context: Optional[Dict[str, Any]]) -> tuple[str, float]:
        """Generate AI response using multiple agents and confidence assessment"""
        
        # Try LLM Gap Detector first for fast responses
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.service_endpoints['llm_gap_detector']}/chat",
                    json={
                        'message': user_message,
                        'mode': 'fast',
                        'conversation_history': conversation['messages'][-10:] if conversation['messages'] else []
                    },
                    timeout=30
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        ai_response = data.get('response', '')
                        confidence = data.get('confidence', 0.5)
                        
                        # If confidence is reasonable, use this response
                        if confidence >= 0.4:
                            return ai_response, confidence
                        else:
                            self.logger.info(f"LLM response has low confidence ({confidence}), trying other agents")
        except Exception as e:
            self.logger.warning(f"LLM Gap Detector failed: {e}")
        
        # Try Neural Engine for more sophisticated reasoning
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.service_endpoints['neural_engine']}/process",
                    json={
                        'query': user_message,
                        'context': context or {},
                        'conversation_history': conversation['messages'][-5:]
                    },
                    timeout=60
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        ai_response = data.get('response', '')
                        confidence = data.get('confidence', 0.3)
                        return ai_response, confidence
        except Exception as e:
            self.logger.warning(f"Neural Engine failed: {e}")
        
        # Try V7 Logic Agent for logical reasoning
        if self._appears_to_be_logical_query(user_message):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.service_endpoints['v7_logic_agent']}/solve",
                        json={
                            'query': user_message,
                            'context': context or {}
                        },
                        timeout=45
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            ai_response = data.get('response', '')
                            confidence = data.get('confidence', 0.6)
                            return ai_response, confidence
            except Exception as e:
                self.logger.warning(f"V7 Logic Agent failed: {e}")
        
        # Fallback response with explicit uncertainty
        uncertainty_responses = [
            "I don't have enough information to provide a reliable answer to that question.",
            "I'm not familiar with that specific topic and would need to learn more about it.",
            "That's beyond my current knowledge base. I'd need additional training to help with that.",
            "I don't have access to current information on that topic.",
            "I'm not certain about that and don't want to provide potentially incorrect information.",
            "I need more information or training data to answer that question properly."
        ]
        
        import random
        fallback_response = random.choice(uncertainty_responses)
        
        return fallback_response, 0.0  # Explicit low confidence for uncertainty
    
    def _appears_to_be_logical_query(self, message: str) -> bool:
        """Check if message appears to be a logical reasoning query"""
        
        logical_indicators = [
            'puzzle', 'logic', 'if then', 'all are', 'none are', 
            'einstein', 'zebra', 'who owns', 'which house',
            'constraint', 'rule', 'given that', 'assume'
        ]
        
        message_lower = message.lower()
        return any(indicator in message_lower for indicator in logical_indicators)
    
    def _determine_confidence_level(self, confidence_score: float, response: str) -> str:
        """Determine confidence level based on score and response content"""
        
        response_lower = response.lower()
        
        # Check for explicit uncertainty phrases
        uncertainty_count = sum(1 for pattern in self.uncertainty_indicators 
                              if re.search(pattern, response_lower))
        
        # Check for hedging language
        hedging_count = sum(1 for pattern in self.low_confidence_patterns 
                           if re.search(pattern, response_lower))
        
        # Adjust confidence based on language patterns
        if uncertainty_count > 0:
            return "unknown"
        elif confidence_score >= 0.8 and hedging_count == 0:
            return "high"
        elif confidence_score >= 0.5 and hedging_count <= 1:
            return "medium"
        elif confidence_score >= 0.2:
            return "low"
        else:
            return "very_low"
    
    async def _analyze_for_knowledge_gaps(self, user_message: str, ai_response: str, 
                                        confidence_score: float) -> tuple[bool, Optional[Dict[str, Any]]]:
        """Analyze interaction for knowledge gaps"""
        
        response_lower = ai_response.lower()
        
        # Check for explicit uncertainty
        explicit_uncertainty = any(re.search(pattern, response_lower) 
                                 for pattern in self.uncertainty_indicators)
        
        # Check for low confidence
        low_confidence = confidence_score < self.confidence_config['uncertainty_trigger_threshold']
        
        gap_detected = explicit_uncertainty or low_confidence
        
        if gap_detected:
            gap_info = {
                'user_query': user_message,
                'ai_response': ai_response,
                'confidence_score': confidence_score,
                'explicit_uncertainty': explicit_uncertainty,
                'gap_type': 'explicit' if explicit_uncertainty else 'confidence_based',
                'detected_at': datetime.now().isoformat()
            }
            return True, gap_info
        
        return False, None
    
    async def _trigger_learning_for_gap(self, gap_info: Dict[str, Any], 
                                      conversation_id: str) -> bool:
        """Trigger learning/LoRA creation for a knowledge gap"""
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.service_endpoints['confidence_lora_creator']}/report_uncertainty",
                    json={
                        'query': gap_info['user_query'],
                        'response': gap_info['ai_response'],
                        'context': {
                            'conversation_id': conversation_id,
                            'gap_type': gap_info['gap_type'],
                            'detected_at': gap_info['detected_at']
                        }
                    },
                    timeout=15
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        lora_triggered = data.get('automatic_lora_triggered', False)
                        
                        if lora_triggered:
                            self.logger.info(f"LoRA creation triggered for gap in conversation {conversation_id}")
                        
                        return lora_triggered
                    else:
                        self.logger.error(f"Failed to report uncertainty: HTTP {response.status}")
                        return False
                        
        except Exception as e:
            self.logger.error(f"Failed to trigger learning for gap: {e}")
            return False
    
    def _generate_user_recommendations(self, confidence_score: float, 
                                     gap_detected: bool, response: str) -> List[str]:
        """Generate recommendations for the user based on confidence"""
        
        recommendations = []
        
        if gap_detected:
            recommendations.append("🔍 I'm learning about this topic and will improve my responses")
            if confidence_score < 0.2:
                recommendations.append("📚 This appears to be a new area for me - I'm gathering information")
        
        if confidence_score < 0.4:
            recommendations.append("⚠️ Please verify this information from authoritative sources")
            recommendations.append("🔄 Consider asking a more specific question")
        
        if confidence_score < 0.6:
            recommendations.append("💡 You might get better results by providing more context")
        
        if any(pattern in response.lower() for pattern in ["i don't know", "i'm not sure"]):
            recommendations.append("🚀 I'm automatically improving my knowledge on this topic")
        
        return recommendations
    
    async def _report_confidence_assessment(self, user_message: str, ai_response: str, 
                                          confidence_score: float, response_time: float,
                                          conversation_id: str):
        """Report confidence assessment to the confidence creator"""
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.service_endpoints['confidence_lora_creator']}/assess_confidence",
                    json={
                        'query': user_message,
                        'response': ai_response,
                        'confidence_score': confidence_score,
                        'response_time': response_time,
                        'model_used': 'ultimate_orchestrator',
                        'context': {
                            'conversation_id': conversation_id,
                            'timestamp': datetime.now().isoformat()
                        }
                    },
                    timeout=10
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('assessment_result', {}).get('confidence_assessment', {}).get('lora_triggered'):
                            self.logger.info(f"Confidence assessment triggered LoRA creation for conversation {conversation_id}")
                    
        except Exception as e:
            self.logger.warning(f"Failed to report confidence assessment: {e}")

# Initialize the orchestrator
orchestrator = UltimateChatOrchestratorWithConfidence()

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint with confidence monitoring"""
    
    logger.info(f"Processing chat message: {request.message[:100]}...")
    
    response = await orchestrator.process_chat_message(request)
    
    return response

@app.post("/chat_simple")
async def chat_simple(message: str, user_id: str = None):
    """Simple chat endpoint for easy integration"""
    
    request = ChatRequest(
        message=message,
        user_id=user_id,
        confidence_reporting=True
    )
    
    response = await orchestrator.process_chat_message(request)
    
    return {
        'response': response.response,
        'confidence_score': response.confidence_score,
        'confidence_level': response.confidence_level,
        'learning_triggered': response.lora_triggered,
        'recommendations': response.recommendations
    }

@app.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history"""
    
    if conversation_id in orchestrator_state['conversations']:
        conversation = orchestrator_state['conversations'][conversation_id]
        return {
            'conversation_id': conversation_id,
            'messages': conversation['messages'],
            'confidence_history': conversation['confidence_history'],
            'started_at': conversation['started_at']
        }
    else:
        raise HTTPException(status_code=404, detail="Conversation not found")

@app.get("/user/{user_id}/sessions")
async def get_user_sessions(user_id: str):
    """Get user's conversation sessions"""
    
    user_conversations = []
    
    for conv_id, conv_data in orchestrator_state['conversations'].items():
        if conv_data.get('user_id') == user_id:
            user_conversations.append({
                'conversation_id': conv_id,
                'started_at': conv_data['started_at'],
                'message_count': len(conv_data['messages']),
                'avg_confidence': sum(h['confidence_score'] for h in conv_data['confidence_history']) / 
                                max(1, len(conv_data['confidence_history']))
            })
    
    return {
        'user_id': user_id,
        'conversations': user_conversations,
        'total_conversations': len(user_conversations)
    }

@app.get("/analytics")
async def get_analytics():
    """Get orchestrator analytics"""
    
    return {
        'orchestrator_stats': orchestrator_state['orchestrator_stats'],
        'confidence_config': orchestrator.confidence_config,
        'active_conversations': len(orchestrator_state['conversations']),
        'learning_requests': len(orchestrator_state['learning_requests']),
        'service_endpoints': orchestrator.service_endpoints
    }

@app.get("/confidence_insights")
async def get_confidence_insights():
    """Get insights about confidence patterns"""
    
    insights = {
        'total_interactions': 0,
        'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0, 'very_low': 0, 'unknown': 0},
        'gap_frequency': 0,
        'learning_triggers': 0,
        'improvement_trend': 'stable'
    }
    
    # Analyze confidence patterns across all conversations
    all_confidence_scores = []
    
    for conversation in orchestrator_state['conversations'].values():
        for entry in conversation['confidence_history']:
            insights['total_interactions'] += 1
            confidence_score = entry['confidence_score']
            all_confidence_scores.append(confidence_score)
            
            if entry['gap_detected']:
                insights['gap_frequency'] += 1
            
            # Categorize confidence level
            if confidence_score >= 0.8:
                insights['confidence_distribution']['high'] += 1
            elif confidence_score >= 0.5:
                insights['confidence_distribution']['medium'] += 1
            elif confidence_score >= 0.2:
                insights['confidence_distribution']['low'] += 1
            elif confidence_score > 0:
                insights['confidence_distribution']['very_low'] += 1
            else:
                insights['confidence_distribution']['unknown'] += 1
    
    # Calculate improvement trend
    if len(all_confidence_scores) >= 10:
        recent_scores = all_confidence_scores[-10:]
        older_scores = all_confidence_scores[-20:-10] if len(all_confidence_scores) >= 20 else all_confidence_scores[:-10]
        
        if older_scores:
            recent_avg = sum(recent_scores) / len(recent_scores)
            older_avg = sum(older_scores) / len(older_scores)
            
            if recent_avg > older_avg * 1.05:
                insights['improvement_trend'] = 'improving'
            elif recent_avg < older_avg * 0.95:
                insights['improvement_trend'] = 'declining'
    
    insights['learning_triggers'] = orchestrator_state['orchestrator_stats']['loras_triggered']
    
    return insights

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    return {
        'status': 'healthy',
        'service': 'ultimate-chat-orchestrator-with-confidence',
        'version': '2.0.0',
        'port': PORT,
        'capabilities': [
            'intelligent_chat_orchestration',
            'real_time_confidence_monitoring',
            'automatic_knowledge_gap_detection',
            'self_triggered_learning',
            'multi_agent_coordination',
            'conversation_awareness',
            'transparency_reporting'
        ],
        'confidence_features': [
            'real_time_confidence_assessment',
            'uncertainty_pattern_detection',
            'automatic_lora_triggering',
            'gap_severity_classification',
            'learning_progress_tracking',
            'user_transparency'
        ],
        'statistics': orchestrator_state['orchestrator_stats'],
        'service_integrations': list(orchestrator.service_endpoints.keys()),
        'confidence_thresholds': orchestrator.confidence_config
    }

@app.get("/")
async def root():
    """Root endpoint"""
    
    return {
        'service': 'Ultimate Chat Orchestrator with Confidence',
        'description': 'AI chat system with automatic knowledge gap detection and LoRA creation',
        'version': '2.0.0',
        'key_features': [
            'Real-time confidence monitoring during conversations',
            'Automatic detection when AI says "I don\'t know"',
            'Self-triggered learning and LoRA creation',
            'Multi-agent orchestration with fallback chains',
            'Transparent confidence reporting to users',
            'Conversation-aware context management',
            'User recommendations based on confidence levels'
        ],
        'workflow': [
            '1. User sends message to chat orchestrator',
            '2. Multiple AI agents attempt to provide response',
            '3. Confidence level assessed in real-time',
            '4. Knowledge gaps detected from uncertainty patterns',
            '5. LoRA creation automatically triggered if needed',
            '6. Response delivered with transparency about confidence',
            '7. Learning progress tracked and improved over time'
        ],
        'confidence_integration': {
            'automatic_gap_detection': True,
            'self_learning_enabled': orchestrator.confidence_config['enable_automatic_lora'],
            'transparency_enabled': orchestrator.confidence_config['enable_transparency'],
            'confidence_thresholds': orchestrator.confidence_config
        },
        'endpoints': {
            'chat': '/chat',
            'chat_simple': '/chat_simple',
            'conversation_history': '/conversation/{conversation_id}',
            'user_sessions': '/user/{user_id}/sessions',
            'analytics': '/analytics',
            'confidence_insights': '/confidence_insights',
            'health': '/health'
        },
        'intelligent_learning_active': True,
        'gap_driven_improvement': True
    }

if __name__ == "__main__":
    logger.info(f"🤖💡 Starting Ultimate Chat Orchestrator with Confidence on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT) 