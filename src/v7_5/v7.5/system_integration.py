"""
LUMINA v7.5 - System Integration Module

This module provides interfaces and fallbacks for integrating with all system components.
It ensures graceful degradation when components are missing or unavailable.
"""

import os
import sys
import time
import json
import logging
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path("logs/v7.5_system.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("LUMINA_v7.5")

# Import Mistral integration
try:
    from src.api.mistral_integration_fixed import MistralIntegration
    MISTRAL_AVAILABLE = True
    logger.info("Mistral integration module loaded successfully")
except ImportError as e:
    logger.warning(f"Mistral integration module not available: {e}")
    MISTRAL_AVAILABLE = False

class SystemIntegration:
    """Main integration class for LUMINA v7.5"""
    
    def __init__(self):
        """Initialize system integration"""
        self.components = {}
        self.mock_mode = False
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize all system components with graceful fallbacks"""
        logger.info("Initializing LUMINA v7.5 system components")
        
        # Get API key from environment or .env file
        api_key = os.environ.get("MISTRAL_API_KEY", "")
        
        # Try to load from .env file if not in environment
        if not api_key:
            try:
                from dotenv import load_dotenv
                load_dotenv(override=True)
                api_key = os.environ.get("MISTRAL_API_KEY", "")
                logger.info(f"API key from dotenv: {'Found' if api_key else 'Not found'}")
            except ImportError:
                logger.warning("dotenv not installed, environment variables used as is")
        
        # Initialize Mistral integration
        if MISTRAL_AVAILABLE and api_key:
            try:
                self.components["mistral_system"] = MistralIntegration(
                    api_key=api_key,
                    model=os.environ.get("LLM_MODEL", "mistral-medium"), 
                    llm_weight=float(os.environ.get("LLM_WEIGHT", "0.7")),
                    nn_weight=float(os.environ.get("NN_WEIGHT", "0.3")),
                    temperature=float(os.environ.get("LLM_TEMPERATURE", "0.7")),
                    top_p=float(os.environ.get("LLM_TOP_P", "0.9")),
                    top_k=int(os.environ.get("LLM_TOP_K", "50"))
                )
                logger.info("Mistral integration initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Mistral integration: {e}")
                self.components["mistral_system"] = MockMistralSystem()
                self.mock_mode = True
        else:
            logger.warning("Mistral integration not available, using mock system")
            self.components["mistral_system"] = MockMistralSystem()
            if not api_key:
                logger.error("Mistral API key not found - please set MISTRAL_API_KEY in environment or .env")
            self.mock_mode = True
        
        # Conversation Flow
        try:
            from conversation_flow import ConversationFlow
            self.components["conversation_flow"] = ConversationFlow()
            logger.info("Loaded Conversation Flow system")
        except ImportError:
            logger.warning("Conversation Flow not available, using mock")
            self.components["conversation_flow"] = MockConversationFlow()
            self.mock_mode = True
        
        # AutoWiki Integration
        try:
            from src.v7.autowiki import AutoWiki
            # Check if Mistral system is available
            mistral_system = self.components.get("mistral_system", None)
            # Initialize AutoWiki with Mistral system if available
            autowiki_dir = Path("data/v7/autowiki")
            autowiki_dir.mkdir(exist_ok=True, parents=True)
            
            mock_autowiki = os.environ.get("AUTOWIKI_MOCK_MODE", "false").lower() == "true"
            auto_fetch = os.environ.get("AUTOWIKI_AUTO_FETCH", "true").lower() == "true"
            
            self.components["autowiki"] = AutoWiki(
                mistral_system=mistral_system,
                data_dir=str(autowiki_dir),
                auto_fetch=auto_fetch,
                mock_mode=mock_autowiki
            )
            logger.info(f"Loaded AutoWiki system (mock_mode: {mock_autowiki}, auto_fetch: {auto_fetch})")
            
            # Start auto-fetching if enabled
            if auto_fetch:
                self.components["autowiki"].start_auto_fetching()
                logger.info("Started AutoWiki auto-fetching")
        except ImportError:
            logger.warning("AutoWiki not available, using mock")
            self.components["autowiki"] = MockAutoWiki()
            self.mock_mode = True
            
        logger.info(f"System initialization complete. Mock mode: {self.mock_mode}")
        
    def process_message(self, message: str) -> Dict[str, Any]:
        """Process a user message through all system components"""
        logger.info(f"Processing message: '{message[:30]}...' (length: {len(message)})")
        
        start_time = time.time()
        
        # Initialize response
        response = {
            "message_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "response": "",
            "insights": [],
            "topics": [],
            "consciousness_level": 0.5,
            "processing_time": 0
        }
        
        # Process with Mistral system if available and not a special command
        if message.startswith("/"):
            # Handle special commands
            command_response = self._handle_command(message)
            if command_response:
                response.update(command_response)
        elif "mistral_system" in self.components:
            try:
                mistral_system = self.components["mistral_system"]
                mistral_result = mistral_system.process_message(message)
                
                if mistral_result.get("combined_response"):
                    response["response"] = mistral_result["combined_response"]
                elif mistral_result.get("llm_response"):
                    response["response"] = mistral_result["llm_response"]
                else:
                    # Fallback to mock response
                    response["response"] = self._generate_mock_response(message)
                    
                # Add insights if available
                if mistral_result.get("nn_processed") and isinstance(mistral_result["nn_processed"], dict):
                    nn_data = mistral_result["nn_processed"]
                    if "insights" in nn_data:
                        response["insights"] = nn_data["insights"]
                    if "topics" in nn_data:
                        response["topics"] = nn_data["topics"]
                    if "consciousness_level" in nn_data:
                        response["consciousness_level"] = nn_data["consciousness_level"]
            except Exception as e:
                logger.error(f"Error processing with Mistral system: {e}")
                response["response"] = f"I'm sorry, I encountered an error processing your message. The system will use backup processing methods."
                response["response"] += "\n\n" + self._generate_mock_response(message)
                response["insights"] = ["Error in primary language processing system"]
        else:
            # Check if this is a knowledge query that should use AutoWiki
            autowiki_keywords = ["what is", "define", "explain", "tell me about", "information on", "learn about"]
            is_knowledge_query = any(message.lower().startswith(keyword) for keyword in autowiki_keywords)
            
            if is_knowledge_query and "autowiki" in self.components:
                # Extract potential topic from knowledge query
                topic = self._extract_topic_from_query(message)
                
                if topic:
                    # Check if AutoWiki already has info on this topic
                    logger.info(f"Knowledge query detected, checking AutoWiki for topic: {topic}")
                    autowiki = self.components["autowiki"]
                    
                    # Add topic to AutoWiki queue for background processing if not already queued
                    autowiki.add_topic(topic)
                    
                    # Try to fetch information immediately
                    topic_info = autowiki.fetch_topic(topic)
                    
                    if topic_info and topic_info.get("success", False):
                        # If we have information available, use it in the response
                        summary = topic_info.get("summary", "")
                        response["response"] = f"Here's what I know about {topic}:\n\n{summary}"
                        
                        # Add source acknowledgment
                        if "url" in topic_info:
                            response["response"] += f"\n\nSource: {topic_info['url']}"
                        
                        # Add topics and insights
                        response["topics"] = [topic]
                        response["insights"] = [f"Information retrieved from AutoWiki knowledge base"]
                    else:
                        # If we don't have info yet, respond accordingly and queue for background processing
                        response["response"] = f"I'm currently learning about {topic}. I've added it to my knowledge queue and will have more information soon."
                        response["topics"] = [topic]
                        response["insights"] = [f"Topic {topic} added to AutoWiki learning queue"]
                else:
                    # Generate regular response if topic extraction failed
                    response["response"] = self._generate_mock_response(message)
                    response["topics"] = self._extract_topics(message)
                    response["insights"] = self._generate_insights_from_response(message, response["response"])
            else:
                # Generate regular response
                response["response"] = self._generate_mock_response(message)
                response["topics"] = self._extract_topics(message)
                response["insights"] = self._generate_insights_from_response(message, response["response"])
        
        # Calculate processing time
        response["processing_time"] = time.time() - start_time
        
        # Process conversation flow if available
        if "conversation_flow" in self.components:
            conversation_context = self.components["conversation_flow"].process_exchange(
                message, response["response"], {"insights": response["insights"]}
            )
            response["conversation_context"] = conversation_context
        
        logger.info(f"Response generated in {response['processing_time']:.2f}s")
        return response
        
    def _handle_command(self, message: str) -> Optional[Dict[str, Any]]:
        """Handle special commands starting with /"""
        command = message.split()[0].lower()
        
        if command == "/status":
            system_state = self.get_system_state()
            return {
                "response": f"System Status:\n- Mock Mode: {system_state['mock_mode']}\n- Components: {len(system_state['components'])}\n- API Connected: {system_state['api_connected']}",
                "insights": ["System status command processed"],
                "topics": ["system", "status"]
            }
        elif command == "/reset":
            if "conversation_flow" in self.components:
                self.components["conversation_flow"].reset()
            return {
                "response": "Conversation context has been reset.",
                "insights": ["Conversation reset command processed"],
                "topics": ["system", "reset"]
            }
        elif command == "/help":
            return {
                "response": "Available commands:\n/status - Show system status\n/reset - Reset conversation context\n/help - Show this help message",
                "insights": ["Help command processed"],
                "topics": ["help", "commands"]
            }
        
        return None
    
    def _extract_topic_from_query(self, query: str) -> str:
        """Extract the main topic from a knowledge query"""
        # Remove common question prefixes
        prefixes = ["what is", "what are", "define", "explain", "tell me about", 
                   "information on", "learn about", "who is", "describe"]
        
        lower_query = query.lower()
        for prefix in prefixes:
            if lower_query.startswith(prefix):
                # Extract the text after the prefix
                topic = query[len(prefix):].strip()
                # Remove punctuation from the end
                if topic and topic[-1] in ".?!":
                    topic = topic[:-1].strip()
                return topic
                
        # If no prefix matched, use the first few words as topic
        words = query.split()
        if len(words) > 2:
            return " ".join(words[:3])
        return query
    
    def _generate_mock_response(self, message: str) -> str:
        """Generate a mock response when language models are not available"""
        # Simple response templates
        templates = [
            "That's an interesting point about {topic}. Have you considered {insight}?",
            "I understand your interest in {topic}. From my perspective, {insight}.",
            "When it comes to {topic}, I think {insight} is important to consider.",
            "I've processed your thoughts on {topic}. One interesting angle is that {insight}.",
            "Thanks for sharing your thoughts on {topic}. It reminds me that {insight}."
        ]
        
        # Extract a potential topic from the message
        topic = self._extract_topic_from_query(message)
        
        # Generate a simple insight
        words = [w for w in message.split() if len(w) > 3]
        if not words:
            words = ["this", "topic", "idea", "concept"]
        
        insight = f"the relationship between {random.choice(words)} and {topic} has many dimensions"
        
        # Select a template and format the response
        template = random.choice(templates)
        response = template.format(topic=topic, insight=insight)
        
        return response
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract potential topics from text"""
        words = text.split()
        # Simple naive topic extraction - just the longest words
        topics = sorted([w.strip('.,?!()[]{}":;') for w in words if len(w) > 5], key=len, reverse=True)
        if not topics:
            # If no long words, use any words
            topics = sorted([w.strip('.,?!()[]{}":;') for w in words if len(w) > 3], key=len, reverse=True)
        
        # Return up to 3 unique topics
        return list(dict.fromkeys(topics))[:3]
    
    def _generate_insights_from_response(self, query: str, response: str) -> List[str]:
        """Generate mock insights from the query and response"""
        # Simple mock insights
        query_words = set([w.lower() for w in query.split() if len(w) > 3])
        response_words = set([w.lower() for w in response.split() if len(w) > 3])
        
        # Find unique words in response not in query
        unique_words = response_words - query_words
        
        if unique_words:
            return [f"The concept of {word} is relevant to this exchange" for word in list(unique_words)[:2]]
        else:
            return ["No specific insights generated for this exchange"]
    
    def get_component(self, name: str) -> Any:
        """Get a specific component by name"""
        return self.components.get(name)
    
    def get_system_state(self) -> Dict[str, Any]:
        """Get the current state of the system"""
        state = {
            "mock_mode": self.mock_mode,
            "components": list(self.components.keys()),
            "timestamp": time.time(),
            "api_connected": False,
            "connection_status": {
                "connected": False,
                "mock_mode": self.mock_mode,
                "last_tested": 0
            }
        }
        
        # Check if Mistral API is connected - with more robust error handling
        if "mistral_system" in self.components:
            try:
                mistral = self.components["mistral_system"]
                
                # Check for API availability
                if hasattr(mistral, "is_available"):
                    mistral_available = mistral.is_available
                    state["api_connected"] = mistral_available
                    
                    # If we have access to detailed connection status, include it
                    if hasattr(mistral, "get_system_stats"):
                        try:
                            stats = mistral.get_system_stats()
                            if "connection_status" in stats:
                                state["connection_status"] = stats["connection_status"]
                            elif "mistral_available" in stats:
                                state["connection_status"]["connected"] = stats["mistral_available"]
                                
                            # Add other useful info from stats
                            state["mistral_stats"] = {
                                "model": stats.get("model", "unknown"),
                                "weights": stats.get("weights", {}),
                                "parameters": stats.get("parameters", {})
                            }
                        except Exception as e:
                            logger.error(f"Error getting Mistral stats: {e}")
                
                # Fallback to basic property check
                elif hasattr(mistral, "client") and mistral.client is not None:
                    state["api_connected"] = True
                    state["connection_status"]["connected"] = True
                    
            except Exception as e:
                logger.error(f"Error checking Mistral connection: {e}")
                state["connection_error"] = str(e)
        
        # Get conversation context if available
        if "conversation_flow" in self.components:
            try:
                state["conversation_context"] = self.components["conversation_flow"].get_conversation_context()
            except Exception as e:
                logger.error(f"Error getting conversation context: {e}")
                state["conversation_error"] = str(e)
        
        # Get AutoWiki status if available
        if "autowiki" in self.components:
            try:
                state["autowiki"] = self.components["autowiki"].get_status()
            except Exception as e:
                logger.error(f"Error getting AutoWiki status: {e}")
                state["autowiki_error"] = str(e)
        
        return state


class MockConversationFlow:
    """Mock implementation of the conversation flow system"""
    
    def __init__(self):
        """Initialize the mock conversation flow"""
        self.exchanges = []
        
    def process_exchange(self, user_message: str, system_response: str, 
                        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a user-system exchange and update conversation context"""
        # Timestamp for this exchange
        timestamp = time.time()
        
        # Create a new exchange record
        exchange = {
            "user_message": user_message,
            "system_response": system_response,
            "timestamp": timestamp,
            "metadata": metadata or {}
        }
        
        # Extract topics from the exchange
        topics = []
        if metadata and "topics" in metadata:
            topics = metadata["topics"]
        else:
            # Simple topic extraction
            words = [w for w in user_message.split() if len(w) > 5]
            topics = words[:3] if words else ["general"]
        
        # Add to exchange record
        exchange["topics"] = topics
        
        # Add to conversation history
        self.exchanges.append(exchange)
        
        # Generate conversation context
        context = {
            "exchange_count": len(self.exchanges),
            "active_topics": self._get_active_topics(),
            "last_exchange_time": timestamp,
            "conversation_duration": timestamp - self.exchanges[0]["timestamp"] if self.exchanges else 0
        }
        
        return context
    
    def get_conversation_context(self) -> Dict[str, Any]:
        """Get the current conversation context"""
        if not self.exchanges:
            return {"exchange_count": 0, "active_topics": [], "last_exchange_time": 0, "conversation_duration": 0}
            
        return {
            "exchange_count": len(self.exchanges),
            "active_topics": self._get_active_topics(),
            "last_exchange_time": self.exchanges[-1]["timestamp"],
            "conversation_duration": time.time() - self.exchanges[0]["timestamp"]
        }
    
    def save_state(self):
        """Save the conversation state"""
        # Mock implementation, doesn't actually save
        pass
    
    def reset(self):
        """Reset the conversation flow"""
        self.exchanges = []
        
    def _get_active_topics(self) -> List[str]:
        """Get active topics from recent exchanges"""
        if not self.exchanges:
            return []
            
        # Get topics from the last 3 exchanges
        recent_exchanges = self.exchanges[-3:]
        all_topics = []
        for exchange in recent_exchanges:
            all_topics.extend(exchange.get("topics", []))
            
        # Return unique topics
        return list(dict.fromkeys(all_topics))


class MockAutoWiki:
    """Mock implementation of the AutoWiki system"""
    
    def __init__(self):
        """Initialize the mock AutoWiki"""
        self.topics = {}
        self.queue = []
        self.fetched_count = 0
        self.auto_fetch_enabled = False
        
        # Pre-populate with a few topics
        self.topics = {
            "neural network": {
                "summary": "A neural network is a computational model inspired by the human brain. It consists of interconnected nodes that process information, allowing the system to learn patterns and make predictions.",
                "url": "https://en.wikipedia.org/wiki/Neural_network",
                "fetched_at": time.time()
            },
            "artificial intelligence": {
                "summary": "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to intelligence displayed by humans or animals. AI applications include advanced web search engines, recommendation systems, and self-driving cars.",
                "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
                "fetched_at": time.time()
            }
        }
    
    def add_topic(self, topic: str) -> bool:
        """Add a topic to the fetch queue"""
        if topic.lower() not in self.topics and topic.lower() not in self.queue:
            self.queue.append(topic.lower())
            return True
        return False
    
    def add_topics(self, topics: List[str]) -> int:
        """Add multiple topics to the fetch queue"""
        added = 0
        for topic in topics:
            if self.add_topic(topic):
                added += 1
        return added
    
    def fetch_topic(self, topic: str) -> Dict[str, Any]:
        """Fetch a topic and return its information"""
        topic_lower = topic.lower()
        
        # Check if we already have this topic
        if topic_lower in self.topics:
            return {
                "success": True,
                "topic": topic,
                "summary": self.topics[topic_lower]["summary"],
                "url": self.topics[topic_lower]["url"],
                "fetched_at": self.topics[topic_lower]["fetched_at"]
            }
            
        # Check if topic is in our pre-defined list
        if topic_lower in ["machine learning", "deep learning", "consciousness"]:
            summary = f"{topic} is a cutting-edge field of study in artificial intelligence. It involves advanced algorithms and computational models that enable computers to learn from and make predictions based on data."
            self.topics[topic_lower] = {
                "summary": summary,
                "url": f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}",
                "fetched_at": time.time()
            }
            self.fetched_count += 1
            
            if topic_lower in self.queue:
                self.queue.remove(topic_lower)
                
            return {
                "success": True,
                "topic": topic,
                "summary": summary,
                "url": f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}",
                "fetched_at": time.time()
            }
            
        # Topic not available yet
        if topic_lower not in self.queue:
            self.queue.append(topic_lower)
            
        return {
            "success": False,
            "topic": topic,
            "message": "Topic not available yet and added to queue"
        }
    
    def process_queue(self, max_items: int = 5) -> Dict[str, Any]:
        """Process items in the queue"""
        processed = 0
        
        for _ in range(min(max_items, len(self.queue))):
            topic = self.queue.pop(0)
            # Just simulate fetching
            if topic not in self.topics:
                self.topics[topic] = {
                    "summary": f"{topic} is an interesting concept with various applications and implications.",
                    "url": f"https://example.com/wiki/{topic.replace(' ', '_')}",
                    "fetched_at": time.time()
                }
                processed += 1
                
        return {"processed": processed, "remaining": len(self.queue)}
    
    def start_auto_fetching(self):
        """Start automatic fetching of queued topics"""
        self.auto_fetch_enabled = True
        
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of AutoWiki"""
        return {
            "topics_available": len(self.topics),
            "queue_size": len(self.queue),
            "topics_fetched": self.fetched_count,
            "auto_fetch_enabled": self.auto_fetch_enabled
        }


class MockMistralSystem:
    """Mock implementation of Mistral system for testing"""
    
    def __init__(self, api_key=None, model="mistral-medium", llm_weight=0.7, nn_weight=0.3,
                 temperature=0.7, top_p=0.9, top_k=50):
        """Initialize mock Mistral system"""
        self.api_key = api_key
        self.model = model
        self.llm_weight = max(0.0, min(1.0, llm_weight))
        self.nn_weight = max(0.0, min(1.0, nn_weight))
        self.temperature = max(0.0, min(1.0, temperature))
        self.top_p = max(0.0, min(1.0, top_p))
        self.top_k = max(1, int(top_k))
        
        # Connection state tracking
        self._connection_tested = True  # Start as connected in mock mode
        self._last_connection_test = time.time()
        
        # Initialize components to match real implementation
        self.client = object()  # Mock client
        self.processor = None   # No processor in mock mode
        
        # Normalize weights
        total_weight = self.llm_weight + self.nn_weight
        if abs(total_weight - 1.0) > 0.001:  # Allow small floating point error
            self.llm_weight /= total_weight
            self.nn_weight /= total_weight
            
        logger.info(f"Initialized mock Mistral system with model {model}")
    
    @property
    def is_available(self) -> bool:
        """Check if Mistral API is available (mock implementation)"""
        # In mock mode, we're always available
        return self.client is not None and self._connection_tested
    
    @property
    def processor_available(self) -> bool:
        """Check if processor is available (mock always returns False)"""
        return self.processor is not None

    def _test_connection(self) -> bool:
        """
        Test the connection (mock implementation)
        
        Returns:
            True (always connected in mock mode)
        """
        self._last_connection_test = time.time()
        self._connection_tested = True
        return True
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics (mock implementation)"""
        return {
            "mistral_available": self.is_available,
            "processor_available": self.processor_available,
            "model": self.model,
            "weights": {
                "llm": self.llm_weight,
                "neural_network": self.nn_weight
            },
            "parameters": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k
            },
            "connection_status": {
                "connected": self.is_available,
                "last_tested": self._last_connection_test,
                "api_key_provided": bool(self.api_key),
                "mock_mode": True
            }
        }
    
    def process_message(self, message: str) -> Dict[str, Any]:
        """Process a message in mock mode"""
        logger.info(f"Mock Mistral system processing message: '{message[:30]}...'")
        logger.info(f"Using parameters: temp={self.temperature}, top_p={self.top_p}, top_k={self.top_k}")
        
        # Create response with mock data
        topics = self._extract_topics(message)
        
        # Create insights based on topics
        insights = [f"Mock insight about {topic}" for topic in topics[:2]]
        if not insights:
            insights = ["No specific topics detected"]
            
        # Add mock neural network processing data
        nn_processed = {
            "insights": insights,
            "topics": topics,
            "consciousness_level": random.uniform(0.4, 0.8),
            "sentiment": random.choice(["neutral", "positive", "negative"]),
            "complexity": random.uniform(0.2, 0.9)
        }
        
        # Generate a mock LLM response
        if "hello" in message.lower() or "hi" in message.lower():
            llm_response = "Hello! I'm LUMINA v7.5, how can I assist you today?"
        elif "?" in message:
            llm_response = f"That's an interesting question. Based on my analysis, I can tell you that there are several factors to consider regarding {', '.join(topics) if topics else 'this topic'}."
        else:
            llm_response = f"I understand you're talking about {', '.join(topics) if topics else 'something interesting'}. I can provide more information if you have specific questions."
        
        # Create combined response with weights
        combined_response = self._combine_responses(llm_response, nn_processed)
        
        return {
            "input": message,
            "llm_response": llm_response,
            "nn_processed": nn_processed,
            "combined_response": combined_response,
            "connection_status": self.is_available,
            "weights": {
                "llm": self.llm_weight,
                "neural_network": self.nn_weight
            },
            "parameters": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k
            }
        }
    
    def _combine_responses(self, llm_response: str, nn_response: Any) -> str:
        """Combine LLM and neural network responses based on weights"""
        # Extract information from neural response if needed
        topics = nn_response.get("topics", []) if isinstance(nn_response, dict) else []
        topic_str = ", ".join(topics) if topics else "the topic"
        
        # If weights heavily favor one side, use that response
        if self.llm_weight > 0.9:
            return llm_response
        if self.nn_weight > 0.9:
            return f"Neural Analysis: Based on my neural processing, {topic_str} has interesting connections to explore."
        
        # Default combination
        return f"{llm_response}\n\nNeural Analysis: I've detected these key topics: {topic_str}."
        
    def adjust_weights(self, llm_weight=None, nn_weight=None, temperature=None, top_p=None, top_k=None):
        """Adjust the weights and parameters (mock implementation)"""
        # Update weights if provided
        if llm_weight is not None:
            self.llm_weight = max(0.0, min(1.0, float(llm_weight)))
        
        if nn_weight is not None:
            self.nn_weight = max(0.0, min(1.0, float(nn_weight)))
        
        # Update LLM parameters if provided
        if temperature is not None:
            self.temperature = max(0.0, min(1.0, float(temperature)))
            
        if top_p is not None:
            self.top_p = max(0.0, min(1.0, float(top_p)))
            
        if top_k is not None:
            self.top_k = max(1, int(top_k))
        
        # Normalize weights to sum to 1.0
        total_weight = self.llm_weight + self.nn_weight
        if abs(total_weight - 1.0) > 0.001:
            self.llm_weight /= total_weight
            self.nn_weight /= total_weight
        
        return {
            "llm_weight": self.llm_weight,
            "nn_weight": self.nn_weight,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k
        }
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text (mock implementation)"""
        words = text.split()
        topics = [w for w in words if len(w) > 4][:3]
        return topics if topics else ["general"]


# Singleton instance
_system_integration = None

def get_system_integration() -> SystemIntegration:
    """Get the singleton system integration instance"""
    global _system_integration
    if _system_integration is None:
        _system_integration = SystemIntegration()
    return _system_integration 