#!/usr/bin/env python3
"""
LoRA System Clients
===================

Client interfaces for all LoRA systems to enable coordinated processing
through the central coordination hub.
"""

import aiohttp
import logging
import asyncio
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseLoRAClient(ABC):
    """Base class for all LoRA system clients"""
    
    def __init__(self, base_url: str, system_name: str):
        self.base_url = base_url.rstrip('/')
        self.system_name = system_name
        self.session = None
        self.timeout = aiohttp.ClientTimeout(total=45)
        
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self.session
    
    async def health_check(self) -> bool:
        """Check if the LoRA system is available"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/health") as response:
                return response.status == 200
        except Exception as e:
            logger.warning(f"{self.system_name} health check failed: {e}")
            return False
    
    @abstractmethod
    async def process_with_coordination(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process request with coordination awareness"""
        pass
    
    @abstractmethod
    async def simple_process(self, query: str) -> Dict[str, Any]:
        """Simple processing for fallback scenarios"""
        pass
    
    async def close(self):
        """Close the aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()

class OptimalLoRARouterClient(BaseLoRAClient):
    """Client for Optimal LoRA Router (Port 5030)"""
    
    def __init__(self, base_url: str = "http://optimal-lora-router:5030"):
        super().__init__(base_url, "optimal_lora_router")
    
    async def process_with_coordination(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process with coordination context"""
        try:
            session = await self._get_session()
            
            # Build LoRA² specific request
            lora_request = {
                "query": request["query"],
                "max_length": 400,
                "context": {
                    "coordination_mode": True,
                    "role": request.get("role_in_coordination", "general"),
                    "coordination_context": request.get("coordination_context", {}),
                    "systems_in_coordination": request.get("coordination_metadata", {}).get("systems_in_coordination", [])
                }
            }
            
            # Try coordination-aware endpoint first
            try:
                async with session.post(f"{self.base_url}/coordinate/process", json=lora_request) as response:
                    if response.status == 200:
                        result = await response.json()
                        return self._format_coordination_response(result, request)
            except:
                pass  # Fall back to standard endpoint
            
            # Fallback to standard generate endpoint
            async with session.post(f"{self.base_url}/generate", json=lora_request) as response:
                if response.status == 200:
                    result = await response.json()
                    return self._format_coordination_response(result, request)
                else:
                    return {"error": f"Request failed with status {response.status}"}
                    
        except Exception as e:
            logger.error(f"Optimal LoRA Router coordination failed: {e}")
            return {"error": str(e), "system": self.system_name}
    
    async def simple_process(self, query: str) -> Dict[str, Any]:
        """Simple processing for fallback"""
        try:
            session = await self._get_session()
            
            request = {"query": query, "max_length": 300}
            
            async with session.post(f"{self.base_url}/generate", json=request) as response:
                if response.status == 200:
                    result = await response.json()
                    return {"response": result.get("response", "No response")}
                else:
                    return {"error": f"Request failed with status {response.status}"}
                    
        except Exception as e:
            return {"error": str(e)}
    
    def _format_coordination_response(self, result: Dict, request: Dict) -> Dict[str, Any]:
        """Format response for coordination framework"""
        return {
            "response": result.get("response", "No response generated"),
            "confidence": result.get("metrics", {}).get("tokens_per_second", 10.0) / 20.0,  # Rough confidence
            "coordination_metadata": {
                "system": self.system_name,
                "role_fulfilled": request.get("role_in_coordination", "general"),
                "lora_selection": result.get("metrics", {}).get("active_loras", []),
                "intent_detected": result.get("query_intent", "unknown")
            },
            "performance_metrics": result.get("metrics", {}),
            "system_specific_data": {
                "lora_weights": result.get("lora_weights", {}),
                "memory_usage": result.get("metrics", {}).get("memory_usage_mb", 0)
            }
        }

class EnhancedPromptClient(BaseLoRAClient):
    """Client for Enhanced Prompt LoRA (Port 8880)"""
    
    def __init__(self, base_url: str = "http://enhanced-prompt-lora:8880"):
        super().__init__(base_url, "enhanced_prompt_lora")
    
    async def process_with_coordination(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process with coordination context"""
        try:
            session = await self._get_session()
            
            # Build Enhanced Prompt LoRA request
            enhance_request = {
                "text": request["query"],
                "target_model": "gemma2:9b",
                "max_tokens": 500,
                "include_system_context": True,
                "context": {
                    "coordination_mode": True,
                    "role": request.get("role_in_coordination", "general"),
                    "coordination_metadata": request.get("coordination_metadata", {})
                }
            }
            
            # Try coordination endpoint first
            try:
                async with session.post(f"{self.base_url}/coordinate/process", json=enhance_request) as response:
                    if response.status == 200:
                        result = await response.json()
                        return self._format_coordination_response(result, request)
            except:
                pass
            
            # Fallback to standard enhance endpoint
            async with session.post(f"{self.base_url}/enhance", json=enhance_request) as response:
                if response.status == 200:
                    result = await response.json()
                    return self._format_coordination_response(result, request)
                else:
                    return {"error": f"Request failed with status {response.status}"}
                    
        except Exception as e:
            logger.error(f"Enhanced Prompt LoRA coordination failed: {e}")
            return {"error": str(e), "system": self.system_name}
    
    async def simple_process(self, query: str) -> Dict[str, Any]:
        """Simple processing for fallback"""
        try:
            session = await self._get_session()
            
            request = {
                "text": query,
                "target_model": "gemma2:9b",
                "max_tokens": 400,
                "include_system_context": False
            }
            
            async with session.post(f"{self.base_url}/enhance", json=request) as response:
                if response.status == 200:
                    result = await response.json()
                    return {"response": result.get("enhanced_content", "No response")}
                else:
                    return {"error": f"Request failed with status {response.status}"}
                    
        except Exception as e:
            return {"error": str(e)}
    
    def _format_coordination_response(self, result: Dict, request: Dict) -> Dict[str, Any]:
        """Format response for coordination framework"""
        return {
            "response": result.get("enhanced_content", "No response generated"),
            "confidence": result.get("confidence", 0.5),
            "coordination_metadata": {
                "system": self.system_name,
                "role_fulfilled": request.get("role_in_coordination", "general"),
                "concept_used": result.get("concept_used", "unknown"),
                "sub_concepts": result.get("sub_concepts", [])
            },
            "performance_metrics": result.get("performance_metrics", {}),
            "system_specific_data": {
                "reasoning_patterns": result.get("reasoning_patterns", []),
                "hallucination_flags": result.get("hallucination_flags", []),
                "system_context_applied": result.get("system_context_applied", False)
            }
        }

class NPUEnhancedClient(BaseLoRAClient):
    """Client for NPU Enhanced LoRA (Port 8881)"""
    
    def __init__(self, base_url: str = "http://enhanced-prompt-lora-npu:8881"):
        super().__init__(base_url, "npu_enhanced_lora")
    
    async def process_with_coordination(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process with NPU acceleration and coordination"""
        try:
            session = await self._get_session()
            
            # Build NPU Enhanced request
            npu_request = {
                "text": request["query"],
                "target_model": "gemma2:9b",
                "max_tokens": 600,
                "use_npu_acceleration": True,
                "include_system_context": True,
                "context": {
                    "coordination_mode": True,
                    "role": request.get("role_in_coordination", "general"),
                    "coordination_context": request.get("coordination_context", {})
                }
            }
            
            # Try coordination endpoint
            try:
                async with session.post(f"{self.base_url}/coordinate/process", json=npu_request) as response:
                    if response.status == 200:
                        result = await response.json()
                        return self._format_coordination_response(result, request)
            except:
                pass
            
            # Fallback to standard enhance endpoint
            async with session.post(f"{self.base_url}/enhance", json=npu_request) as response:
                if response.status == 200:
                    result = await response.json()
                    return self._format_coordination_response(result, request)
                else:
                    return {"error": f"Request failed with status {response.status}"}
                    
        except Exception as e:
            logger.error(f"NPU Enhanced LoRA coordination failed: {e}")
            return {"error": str(e), "system": self.system_name}
    
    async def simple_process(self, query: str) -> Dict[str, Any]:
        """Simple processing with NPU acceleration"""
        try:
            session = await self._get_session()
            
            request = {
                "text": query,
                "target_model": "gemma2:9b",
                "max_tokens": 500,
                "use_npu_acceleration": True
            }
            
            async with session.post(f"{self.base_url}/enhance", json=request) as response:
                if response.status == 200:
                    result = await response.json()
                    return {"response": result.get("enhanced_content", "No response")}
                else:
                    return {"error": f"Request failed with status {response.status}"}
                    
        except Exception as e:
            return {"error": str(e)}
    
    def _format_coordination_response(self, result: Dict, request: Dict) -> Dict[str, Any]:
        """Format NPU response for coordination framework"""
        return {
            "response": result.get("enhanced_content", "No response generated"),
            "confidence": result.get("confidence", 0.5),
            "coordination_metadata": {
                "system": self.system_name,
                "role_fulfilled": request.get("role_in_coordination", "general"),
                "npu_acceleration_used": result.get("npu_acceleration_used", False),
                "concept_used": result.get("concept_used", "unknown")
            },
            "performance_metrics": result.get("performance_metrics", {}),
            "system_specific_data": {
                "npu_performance": result.get("npu_performance", {}),
                "processing_time_ms": result.get("processing_time_ms", 0),
                "cache_hit": result.get("cache_hit", False)
            }
        }

class NPUAdapterClient(BaseLoRAClient):
    """Client for NPU Adapter Selector (Port 5020)"""
    
    def __init__(self, base_url: str = "http://npu-adapter-selector:5020"):
        super().__init__(base_url, "npu_adapter_selector")
    
    async def process_with_coordination(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process with adapter selection coordination"""
        try:
            session = await self._get_session()
            
            # Build adapter selection request
            adapter_request = {
                "query": request["query"],
                "context": {
                    "coordination_mode": True,
                    "role": request.get("role_in_coordination", "general"),
                    "available_adapters": request.get("coordination_context", {}).get("preferred_adapters", [])
                }
            }
            
            # Try coordination endpoint
            try:
                async with session.post(f"{self.base_url}/coordinate/process", json=adapter_request) as response:
                    if response.status == 200:
                        result = await response.json()
                        return self._format_coordination_response(result, request)
            except:
                pass
            
            # Fallback to selection endpoint
            async with session.post(f"{self.base_url}/select_adapter", json=adapter_request) as response:
                if response.status == 200:
                    result = await response.json()
                    return self._format_coordination_response(result, request)
                else:
                    return {"error": f"Request failed with status {response.status}"}
                    
        except Exception as e:
            logger.error(f"NPU Adapter Selector coordination failed: {e}")
            return {"error": str(e), "system": self.system_name}
    
    async def simple_process(self, query: str) -> Dict[str, Any]:
        """Simple adapter selection"""
        try:
            session = await self._get_session()
            
            request = {"query": query}
            
            async with session.post(f"{self.base_url}/select_adapter", json=request) as response:
                if response.status == 200:
                    result = await response.json()
                    return {"response": result.get("response", "Adapter selected")}
                else:
                    return {"error": f"Request failed with status {response.status}"}
                    
        except Exception as e:
            return {"error": str(e)}
    
    def _format_coordination_response(self, result: Dict, request: Dict) -> Dict[str, Any]:
        """Format adapter selection response for coordination"""
        return {
            "response": result.get("response", "Adapter selection completed"),
            "confidence": result.get("confidence", 0.8),
            "coordination_metadata": {
                "system": self.system_name,
                "role_fulfilled": request.get("role_in_coordination", "general"),
                "selected_adapter": result.get("selected_adapter", "unknown"),
                "adapter_confidence": result.get("adapter_confidence", 0.5)
            },
            "performance_metrics": result.get("performance_metrics", {}),
            "system_specific_data": {
                "available_adapters": result.get("available_adapters", []),
                "selection_reasoning": result.get("reasoning", []),
                "alternatives": result.get("alternatives", [])
            }
        }

class JarvisChatClient(BaseLoRAClient):
    """Client for Jarvis Chat Interface (Port 5010)"""
    
    def __init__(self, base_url: str = "http://jarvis-chat:5010"):
        super().__init__(base_url, "jarvis_chat")
    
    async def process_with_coordination(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process with Jarvis chat coordination"""
        try:
            session = await self._get_session()
            
            # Build Jarvis chat request
            chat_request = {
                "message": request["query"],
                "context": {
                    "coordination_mode": True,
                    "role": request.get("role_in_coordination", "general"),
                    "systems_active": request.get("coordination_metadata", {}).get("systems_in_coordination", [])
                }
            }
            
            # Try coordination endpoint
            try:
                async with session.post(f"{self.base_url}/coordinate/chat", json=chat_request) as response:
                    if response.status == 200:
                        result = await response.json()
                        return self._format_coordination_response(result, request)
            except:
                pass
            
            # Fallback to standard chat endpoint
            async with session.post(f"{self.base_url}/chat", json=chat_request) as response:
                if response.status == 200:
                    result = await response.json()
                    return self._format_coordination_response(result, request)
                else:
                    return {"error": f"Request failed with status {response.status}"}
                    
        except Exception as e:
            logger.error(f"Jarvis Chat coordination failed: {e}")
            return {"error": str(e), "system": self.system_name}
    
    async def simple_process(self, query: str) -> Dict[str, Any]:
        """Simple chat processing"""
        try:
            session = await self._get_session()
            
            request = {"message": query}
            
            async with session.post(f"{self.base_url}/chat", json=request) as response:
                if response.status == 200:
                    result = await response.json()
                    return {"response": result.get("response", "No response")}
                else:
                    return {"error": f"Request failed with status {response.status}"}
                    
        except Exception as e:
            return {"error": str(e)}
    
    def _format_coordination_response(self, result: Dict, request: Dict) -> Dict[str, Any]:
        """Format Jarvis chat response for coordination"""
        return {
            "response": result.get("response", "No response generated"),
            "confidence": result.get("confidence", 0.7),
            "coordination_metadata": {
                "system": self.system_name,
                "role_fulfilled": request.get("role_in_coordination", "general"),
                "chat_mode": result.get("chat_mode", "standard"),
                "adapters_used": result.get("adapters_used", [])
            },
            "performance_metrics": result.get("performance_metrics", {}),
            "system_specific_data": {
                "conversation_context": result.get("conversation_context", {}),
                "jarvis_personality": result.get("personality_applied", "default"),
                "response_type": result.get("response_type", "conversational")
            }
        } 