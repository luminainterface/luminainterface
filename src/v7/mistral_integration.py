#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mistral AI Integration Module for V7

This module provides integration with Mistral AI's language models,
including an autowiki learning system that builds a dictionary of knowledge.
"""

import os
import json
import time
import logging
import random
import hashlib
import threading
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import requests
from requests.exceptions import RequestException
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class MistralIntegration:
    """
    Integration with Mistral AI language models with autowiki learning capabilities.
    
    This class provides methods to:
    - Connect to Mistral AI API
    - Process messages and generate responses
    - Maintain a learning dictionary via autowiki
    - Track usage metrics
    """
    
    # Available Mistral models
    AVAILABLE_MODELS = [
        "mistral-tiny",
        "mistral-small",
        "mistral-medium",
        "mistral-large-latest"
    ]
    
    # Endpoint for Mistral API
    API_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "mistral-medium",
        mock_mode: bool = False,
        learning_enabled: bool = True,
        learning_dict_path: str = "data/mistral_learning.json",
        max_memory_entries: int = 100,
        learning_save_interval: int = 10,
    ):
        """
        Initialize the Mistral integration.
        
        Args:
            api_key: Mistral API key (optional if using mock_mode)
            model: Mistral model to use
            mock_mode: If True, generate mock responses instead of calling the API
            learning_enabled: If True, enable the autowiki learning functionality
            learning_dict_path: Path to save/load the learning dictionary
            max_memory_entries: Maximum number of memory entries to keep
            learning_save_interval: How often to save the learning dictionary (in minutes)
        """
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        self.model = model if model in self.AVAILABLE_MODELS else "mistral-medium"
        self.mock_mode = mock_mode or not api_key
        self.learning_enabled = learning_enabled
        self.learning_dict_path = learning_dict_path
        self.max_memory_entries = max_memory_entries
        
        # Set up data directory and learning dictionary path
        self.data_dir = Path(__file__).parent / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        if learning_dict_path:
            self.learning_dict_path = Path(learning_dict_path)
        else:
            self.learning_dict_path = self.data_dir / "mistral_learning_dict.json"
            
        # Initialize metrics
        self.metrics = {
            "api_calls": 0,
            "tokens_used": 0,
            "tokens_prompt": 0,
            "tokens_completion": 0,
            "learning_dict_size": 0,
            "autowiki_entries": 0,
            "conversation_memory_entries": 0
        }
        
        # Initialize learning dictionary and conversation memory
        self.learning_dict = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "updated": datetime.now().isoformat(),
                "version": "1.0",
                "model": self.model,
            },
            "autowiki": {},
            "conversation_memory": [],
        }
        
        # Load existing learning dictionary if available
        self._load_learning_dictionary()
        
        # Start background thread to periodically save the learning dictionary
        if learning_enabled and learning_save_interval > 0:
            self._start_background_save(learning_save_interval)
        
        logger.info(f"Initialized MistralIntegration with model: {self.model}")
        if self.mock_mode:
            logger.warning("Running in mock mode - API calls will be simulated")
    
    def _load_learning_dictionary(self) -> None:
        """Load the learning dictionary from disk if it exists."""
        if not self.learning_enabled or not self.learning_dict_path:
            return
        
        try:
            if os.path.exists(self.learning_dict_path):
                with open(self.learning_dict_path, 'r') as f:
                    loaded_dict = json.load(f)
                    self.learning_dict = loaded_dict
                    logger.info(f"Loaded learning dictionary from {self.learning_dict_path}")
                    
                    # Update metadata
                    self.learning_dict["metadata"]["model"] = self.model
                    
                    # Count autowiki entries
                    autowiki_count = len(self.learning_dict.get("autowiki", {}))
                    logger.info(f"Loaded {autowiki_count} autowiki entries")
                    
                    # Update metrics
                    self.metrics["learning_dict_size"] = len(self.learning_dict.get("autowiki", {}))
                    self.metrics["autowiki_entries"] = autowiki_count
            else:
                logger.info("No learning dictionary found, creating a new one")
        except Exception as e:
            logger.error(f"Error loading learning dictionary: {str(e)}")
            # Create a new dictionary if loading fails
            self._reset_learning_dictionary()
    
    def _reset_learning_dictionary(self) -> None:
        """Reset the learning dictionary to its initial state."""
        self.learning_dict = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "updated": datetime.now().isoformat(),
                "version": "1.0",
                "model": self.model,
            },
            "autowiki": {},
            "conversation_memory": [],
        }
        logger.info("Reset learning dictionary to initial state")
    
    def save_learning_dictionary(self) -> bool:
        """
        Save the learning dictionary to disk.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.learning_enabled or not self.learning_dict_path:
            return False
        
        try:
            # Update metadata
            self.learning_dict["metadata"]["updated"] = datetime.now().isoformat()
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.learning_dict_path), exist_ok=True)
            
            # Save to disk
            with open(self.learning_dict_path, 'w') as f:
                json.dump(self.learning_dict, f, indent=2)
            
            logger.info(f"Saved learning dictionary to {self.learning_dict_path}")
            
            # Update metrics
            self.metrics["learning_dict_size"] = len(self.learning_dict.get("autowiki", {}))
            self.metrics["autowiki_entries"] = len(self.learning_dict.get("autowiki", {}))
            
            return True
        except Exception as e:
            logger.error(f"Error saving learning dictionary: {str(e)}")
            return False
    
    def _start_background_save(self, interval_minutes: int) -> None:
        """
        Start a background thread to periodically save the learning dictionary.
        
        Args:
            interval_minutes: How often to save the dictionary (in minutes)
        """
        def _save_periodically():
            while True:
                time.sleep(interval_minutes * 60)
                if self.learning_enabled:
                    self.save_learning_dictionary()
        
        save_thread = threading.Thread(target=_save_periodically, daemon=True)
        save_thread.start()
        logger.info(f"Started background save thread (interval: {interval_minutes} minutes)")
    
    def add_autowiki_entry(
        self, 
        topic: str, 
        content: str, 
        source: Optional[str] = None
    ) -> bool:
        """
        Add or update an entry in the autowiki.
        
        Args:
            topic: The topic/title for the entry
            content: The content/knowledge to store
            source: Optional source/attribution for the knowledge
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.learning_enabled:
            logger.warning("Learning is disabled - cannot add autowiki entry")
            return False
        
        try:
            # Ensure autowiki dictionary exists
            if "autowiki" not in self.learning_dict:
                self.learning_dict["autowiki"] = {}
            
            # Check if entry already exists
            is_update = topic.lower() in {k.lower() for k in self.learning_dict["autowiki"].keys()}
            
            # Find exact key if it's a case-insensitive match
            if is_update:
                existing_key = next(k for k in self.learning_dict["autowiki"].keys() 
                                   if k.lower() == topic.lower())
                
                # Get existing entry and update it
                existing_entry = self.learning_dict["autowiki"][existing_key]
                
                # Use the original topic name to maintain case
                entry = {
                    "content": content,
                    "sources": existing_entry.get("sources", []),
                    "created": existing_entry.get("created", datetime.now().isoformat()),
                    "updated": datetime.now().isoformat(),
                }
                
                # Add source if provided and not already in sources
                if source and source not in entry["sources"]:
                    entry["sources"].append(source)
                
                # Update the entry using the original key
                self.learning_dict["autowiki"][existing_key] = entry
                logger.info(f"Updated autowiki entry: {existing_key}")
            else:
                # Create new entry
                entry = {
                    "content": content,
                    "sources": [source] if source else [],
                    "created": datetime.now().isoformat(),
                    "updated": datetime.now().isoformat(),
                }
                
                self.learning_dict["autowiki"][topic] = entry
                logger.info(f"Added new autowiki entry: {topic}")
            
            # Automatically save the dictionary
            self.save_learning_dictionary()
            return True
        except Exception as e:
            logger.error(f"Error adding autowiki entry: {str(e)}")
            return False
    
    def retrieve_autowiki(self, topic: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve an entry from the autowiki by topic.
        
        Args:
            topic: The topic to retrieve
            
        Returns:
            Optional[Dict[str, Any]]: The entry if found, None otherwise
        """
        if not self.learning_enabled or "autowiki" not in self.learning_dict:
            return None
        
        # Check for exact match first
        if topic in self.learning_dict["autowiki"]:
            return self.learning_dict["autowiki"][topic]
        
        # Try case-insensitive match
        for key in self.learning_dict["autowiki"].keys():
            if key.lower() == topic.lower():
                return self.learning_dict["autowiki"][key]
        
        return None
    
    def delete_autowiki_entry(self, topic: str) -> bool:
        """
        Delete an entry from the autowiki.
        
        Args:
            topic: The topic to delete
            
        Returns:
            bool: True if deleted, False if not found or error
        """
        if not self.learning_enabled or "autowiki" not in self.learning_dict:
            return False
        
        # Check for exact match first
        if topic in self.learning_dict["autowiki"]:
            del self.learning_dict["autowiki"][topic]
            self.save_learning_dictionary()
            logger.info(f"Deleted autowiki entry: {topic}")
            return True
        
        # Try case-insensitive match
        for key in list(self.learning_dict["autowiki"].keys()):
            if key.lower() == topic.lower():
                del self.learning_dict["autowiki"][key]
                self.save_learning_dictionary()
                logger.info(f"Deleted autowiki entry: {key}")
                return True
        
        logger.warning(f"Autowiki entry not found: {topic}")
        return False
    
    def get_all_autowiki_topics(self) -> List[str]:
        """
        Get all topics in the autowiki.
        
        Returns:
            List[str]: List of all topics
        """
        if not self.learning_enabled or "autowiki" not in self.learning_dict:
            return []
        
        return list(self.learning_dict["autowiki"].keys())
    
    def search_autowiki(self, query: str) -> List[Dict[str, Any]]:
        """
        Search the autowiki for entries matching the query.
        
        Args:
            query: The search query
            
        Returns:
            List[Dict[str, Any]]: List of matching entries with their topics
        """
        if not self.learning_enabled or "autowiki" not in self.learning_dict:
            return []
        
        results = []
        query = query.lower()
        
        for topic, entry in self.learning_dict["autowiki"].items():
            content = entry.get("content", "").lower()
            
            if query in topic.lower() or query in content:
                results.append({
                    "topic": topic,
                    "entry": entry,
                    "relevance": self._calculate_relevance(query, topic, content)
                })
        
        # Sort by relevance score (higher is better)
        results.sort(key=lambda x: x["relevance"], reverse=True)
        
        return results
    
    def _calculate_relevance(self, query: str, topic: str, content: str) -> float:
        """
        Calculate a relevance score for a search result.
        
        Args:
            query: The search query
            topic: The topic of the entry
            content: The content of the entry
            
        Returns:
            float: Relevance score (higher is better)
        """
        # Simple relevance calculation based on exact matches
        # More sophisticated algorithms could be implemented here
        relevance = 0.0
        
        # Topic matches are weighted higher
        if query in topic.lower():
            relevance += 3.0
            # Exact topic match gets highest score
            if query == topic.lower():
                relevance += 5.0
        
        # Content matches
        if query in content:
            relevance += 1.0
            # Count occurrences in content
            relevance += 0.1 * content.count(query)
        
        return relevance
    
    def add_to_conversation_memory(
        self, 
        message: str, 
        response: str, 
        system_prompt: Optional[str] = None
    ) -> None:
        """
        Add a conversation to memory.
        
        Args:
            message: The user message
            response: The model response
            system_prompt: Optional system prompt used
        """
        if not self.learning_enabled:
            return
        
        # Ensure conversation_memory exists
        if "conversation_memory" not in self.learning_dict:
            self.learning_dict["conversation_memory"] = []
        
        # Add conversation to memory
        conversation = {
            "timestamp": datetime.now().isoformat(),
            "system_prompt": system_prompt,
            "message": message,
            "response": response,
            "model": self.model,
        }
        
        self.learning_dict["conversation_memory"].append(conversation)
        
        # Limit conversation memory size
        if (len(self.learning_dict["conversation_memory"]) > self.max_memory_entries and
            self.max_memory_entries > 0):
            # Remove oldest conversations
            self.learning_dict["conversation_memory"] = (
                self.learning_dict["conversation_memory"][-self.max_memory_entries:]
            )
    
    def process_message(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        include_autowiki: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a message with the Mistral model.
        
        Args:
            message: The message to process
            system_prompt: Optional system prompt
            temperature: Temperature parameter (0.0-1.0)
            max_tokens: Maximum number of tokens to generate
            include_autowiki: Whether to include autowiki entries as context
            
        Returns:
            Dict[str, Any]: Response dictionary with model, response, etc.
        """
        if not message:
            return {"error": "Empty message", "response": "I received an empty message."}
        
        try:
            # Use mock response if in mock mode
            if self.mock_mode:
                response = self._generate_mock_response(message, include_autowiki)
                
                # Add to conversation memory
                self.add_to_conversation_memory(message, response, system_prompt)
                
                return {
                    "model": f"{self.model} (mock)",
                    "response": response,
                    "tokens_used": 0,
                    "tokens_prompt": 0,
                    "tokens_completion": 0,
                }
            
            # Check if API key is available
            if not self.api_key:
                return {
                    "error": "API key not available",
                    "response": "Error: Mistral API key not provided. Please set up your API key in the settings."
                }
            
            # Prepare messages for API call
            messages = []
            
            # Add system message if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Add autowiki context if enabled
            if include_autowiki and self.learning_enabled:
                autowiki_context = self._get_autowiki_context(message)
                if autowiki_context:
                    messages.append({"role": "system", "content": autowiki_context})
            
            # Add user message
            messages.append({"role": "user", "content": message})
            
            # Call Mistral API
            api_response = self._call_mistral_api(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            if "error" in api_response:
                return api_response
            
            # Update metrics
            self.metrics["api_calls"] += 1
            self.metrics["tokens_used"] += api_response.get("usage", {}).get("total_tokens", 0)
            self.metrics["tokens_prompt"] += api_response.get("usage", {}).get("prompt_tokens", 0)
            self.metrics["tokens_completion"] += api_response.get("usage", {}).get("completion_tokens", 0)
            
            # Extract response
            response_text = api_response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Add to conversation memory
            self.add_to_conversation_memory(message, response_text, system_prompt)
            
            return {
                "model": self.model,
                "response": response_text,
                "tokens_used": api_response.get("usage", {}).get("total_tokens", 0),
                "tokens_prompt": api_response.get("usage", {}).get("prompt_tokens", 0),
                "tokens_completion": api_response.get("usage", {}).get("completion_tokens", 0),
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return {
                "error": str(e),
                "response": f"Error processing your message: {str(e)}"
            }
    
    def _get_autowiki_context(self, message: str) -> Optional[str]:
        """
        Get relevant autowiki entries to use as context.
        
        Args:
            message: The user message to find relevant entries for
            
        Returns:
            Optional[str]: Formatted context string or None if no relevant entries
        """
        if not self.learning_enabled or "autowiki" not in self.learning_dict:
            return None
        
        # Search autowiki for relevant entries
        search_results = self.search_autowiki(message.lower())
        
        # Take top N most relevant results
        top_results = search_results[:3]  # Limit to 3 most relevant entries
        
        if not top_results:
            return None
        
        # Format context string
        context_parts = ["Here is some relevant information from my knowledge base:"]
        
        for result in top_results:
            topic = result["topic"]
            entry = result["entry"]
            content = entry.get("content", "")
            sources = entry.get("sources", [])
            
            context_parts.append(f"\nTOPIC: {topic}")
            context_parts.append(f"CONTENT: {content}")
            if sources:
                context_parts.append(f"SOURCE: {', '.join(sources)}")
            context_parts.append("---")
        
        context_parts.append("\nPlease use this information to help answer the user's question if relevant.")
        
        return "\n".join(context_parts)
    
    def _call_mistral_api(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """
        Call the Mistral API.
        
        Args:
            messages: List of message objects
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dict[str, Any]: API response
        """
        url = "https://api.mistral.ai/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            logger.error(f"API error: {str(e)}")
            return {"error": str(e), "response": f"API Error: {str(e)}"}
    
    def _generate_mock_response(
        self, 
        message: str, 
        include_autowiki: bool
    ) -> str:
        """
        Generate a mock response for testing without API calls.
        
        Args:
            message: The user message
            include_autowiki: Whether autowiki was included
            
        Returns:
            str: Mock response
        """
        # If autowiki was included and learning is enabled, try to use it
        if include_autowiki and self.learning_enabled:
            # Search for relevant entries
            search_results = self.search_autowiki(message.lower())
            
            if search_results:
                # Use top result to generate a response
                top_result = search_results[0]
                topic = top_result["topic"]
                content = top_result["entry"].get("content", "")
                
                responses = [
                    f"Based on my knowledge about {topic}, {content}",
                    f"According to my information on {topic}, {content}",
                    f"I found some relevant information about {topic}. {content}",
                ]
                return random.choice(responses)
        
        # Generic mock responses
        generic_responses = [
            "This is a mock response in development mode. In production, this would connect to the Mistral AI API.",
            "I'm currently in mock mode. To get real responses, disable mock mode in settings and provide an API key.",
            "Mock mode is active. This response is generated locally without making API calls.",
            f"I received your message: '{message}'. This is a simulated response from the mock system.",
            "When connected to the Mistral API, I would provide an accurate response to your query.",
        ]
        
        # Questions and answers for common topics
        if "neural network" in message.lower():
            return "Neural networks are computing systems inspired by biological neural networks. They consist of layers of interconnected nodes or 'neurons' that process and transform input data to generate outputs."
        
        if "consciousness" in message.lower():
            return "Consciousness is the state of being awake and aware of one's surroundings, thoughts, and identity. It remains one of the most profound mysteries in neuroscience and philosophy of mind."
        
        if "mistral" in message.lower():
            return "Mistral AI is a company that develops large language models and offers an API for accessing these models. Their models include mistral-tiny, mistral-small, mistral-medium, and mistral-large, with varying capabilities and prices."
        
        if "autowiki" in message.lower():
            return "The autowiki feature allows the system to learn and store information that can be retrieved later. It functions like a knowledge base that grows and improves over time as new information is added."
        
        # Return a random generic response
        return random.choice(generic_responses)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get usage metrics.
        
        Returns:
            Dict[str, Any]: Metrics dictionary
        """
        metrics = {
            "api_calls": self.metrics["api_calls"],
            "tokens_used": self.metrics["tokens_used"],
            "tokens_prompt": self.metrics["tokens_prompt"],
            "tokens_completion": self.metrics["tokens_completion"],
            "learning_enabled": self.learning_enabled,
            "learning_dict_path": self.learning_dict_path,
            "mock_mode": self.mock_mode,
            "model": self.model,
        }
        
        # Add learning dictionary stats if enabled
        if self.learning_enabled:
            # Dictionary size (rough estimate)
            try:
                dict_size = len(json.dumps(self.learning_dict).encode('utf-8'))
                metrics["learning_dict_size"] = dict_size
            except Exception:
                metrics["learning_dict_size"] = 0
            
            # Count autowiki entries
            metrics["autowiki_entries"] = len(self.learning_dict.get("autowiki", {}))
            
            # Count conversation memory entries
            metrics["conversation_memory_entries"] = len(self.learning_dict.get("conversation_memory", []))
        
        return metrics

if __name__ == "__main__":
    # Example usage
    mistral = MistralIntegration(mock_mode=True)
    
    # Add a test entry
    mistral.add_autowiki_entry(
        topic="Neural Networks",
        content="Neural networks are computing systems inspired by the human brain. They consist of layers of interconnected nodes that process and transform data.",
        source="Example Source"
    )
    
    # Process a message
    response = mistral.process_message(
        message="Tell me about neural networks",
        include_autowiki=True
    )
    
    print(f"Response: {response['response']}")
    
    # Get metrics
    metrics = mistral.get_metrics()
    print(f"Metrics: {metrics}")
