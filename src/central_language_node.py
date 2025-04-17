#!/usr/bin/env python3
"""
Central Language Node

This module serves as the central integration point for all language-related components
in the Lumina Neural Network system, including:

- Language Memory System
- English Language Trainer
- Conversation Memory
- Language Memory Synthesis
- LLM Enhancement
- Neural Network Integration
- Memory API
- V10 Conscious Mirror Language
- Neural Linguistic Processor

It provides a unified interface for v5-v10 evolution and supports the Conscious Mirror capabilities.
"""

import os
import sys
import logging
import importlib
import json
from datetime import datetime
from pathlib import Path
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("central_language_node.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("central_language_node")

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))


class CentralLanguageNode:
    """
    Central integration point for all language-related components
    
    This class serves as the unified interface to all language, memory, and neural network
    components in the Lumina system, supporting the v5-v10 evolution roadmap.
    
    Key features:
    - Dynamic component discovery and integration
    - Unified interface for all language operations
    - Cross-component memory synthesis
    - Neural network and LLM integration
    - Support for v5-v10 consciousness features
    - Extensible architecture for future enhancements
    """
    
    def __init__(self, config_path: Optional[str] = None, components: Optional[Dict[str, Any]] = None):
        """
        Initialize the Central Language Node
        
        Args:
            config_path: Path to configuration file (optional)
            components: Pre-initialized components to use (optional)
        """
        logger.info("Initializing Central Language Node")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize component registry
        self.components = components or {}
        self.component_status = {}
        
        # Initialize metrics
        self.metrics = {
            "initialization_time": datetime.now().isoformat(),
            "operations_count": 0,
            "component_usage": {},
        }
        
        # Discover and initialize components
        self._discover_components()
        
        # Initialize integration points
        self._initialize_integrations()
        
        # Register with v5-v10 systems if available
        self._register_with_v10_system()
        
        logger.info(f"Central Language Node initialized with {len(self.components)} components")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            "data_path": "data/memory",
            "component_priorities": {
                "language_memory": 100,
                "conversation_memory": 90,
                "english_language_trainer": 80,
                "memory_synthesis": 70,
                "llm_enhancement": 60,
                "neural_network": 50,
                "memory_api": 40
            },
            "enable_v5_integration": True,
            "enable_v10_features": True,
            "consciousness_level": 2,  # 0-5 scale
            "memory_retention_days": 365,
            "auto_synthesis_interval_hours": 24
        }
        
        if not config_path:
            logger.info("Using default configuration")
            return default_config
        
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                config = {**default_config, **user_config}
                logger.info(f"Loaded configuration from {config_path}")
                return config
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {str(e)}")
            logger.info("Using default configuration")
            return default_config
    
    def _discover_components(self):
        """Discover and initialize all available language components"""
        logger.info("Discovering language components")
        
        # Define components to discover
        component_modules = [
            # Language Memory components
            {"name": "language_memory", "module": "language.language_memory", "class": "LanguageMemory"},
            {"name": "language_memory_integration", "module": "language.language_memory_integration", "function": "integrate_language_memory"},
            
            # Memory system components
            {"name": "conversation_memory", "module": "conversation_memory", "class": "ConversationMemory"},
            {"name": "memory_synthesis", "module": "memory.memory_synthesis", "class": "MemorySynthesis"},
            {"name": "conversation_language_bridge", "module": "memory.conversation_language_bridge", "class": "ConversationLanguageBridge"},
            
            # Synthesis and training components
            {"name": "language_memory_synthesis", "module": "language_memory_synthesis_integration", "class": "LanguageMemorySynthesisIntegration"},
            {"name": "english_language_trainer", "module": "english_language_trainer", "class": "EnglishLanguageTrainer"},
            
            # API components
            {"name": "memory_api", "module": "memory_api", "class": "MemoryAPI"},
            {"name": "memory_api_server", "module": "memory_api_server", "class": "MemoryAPIServer"},
            {"name": "memory_api_client", "module": "memory_api_client", "class": "MemoryAPIClient"},
            
            # LLM enhancement
            {"name": "llm_enhancement", "module": "enhance_llm_prompt", "class": "LLMPromptEnhancer"},
            
            # V5 integration 
            {"name": "v5_language_integration", "module": "v5.language_memory_integration", "class": "LanguageMemoryIntegrationPlugin"},
            {"name": "v5_neural_state", "module": "v5.neural_state_plugin", "class": "NeuralStatePlugin"},
            
            # V5 bridge (if PySide6 is available)
            {"name": "v5_bridge", "module": "v5_integration.visualization_bridge", "function": "get_visualization_bridge"},
            
            # V10 Conscious Mirror Language
            {"name": "conscious_mirror_language", "module": "v10.conscious_mirror_language", "function": "get_conscious_mirror"},
            {"name": "v7_node_consciousness", "module": "v7.node_consciousness", "class": "NodeConsciousness"},
            {"name": "v9_mirror_consciousness", "module": "v9.mirror_consciousness", "class": "MirrorConsciousness"},
            
            # Advanced Neural Linguistic Processing
            {"name": "neural_linguistic_processor", "module": "language.neural_linguistic_processor", "function": "get_neural_linguistic_processor"}
        ]
        
        # Try to import and initialize each component
        for component_info in component_modules:
            component_name = component_info["name"]
            module_name = component_info["module"]
            
            try:
                # Import the module
                module = importlib.import_module(f"src.{module_name}")
                logger.info(f"Successfully imported {module_name}")
                
                # Initialize based on component type
                if "class" in component_info:
                    # Class-based component
                    class_name = component_info["class"]
                    component_class = getattr(module, class_name)
                    
                    # Initialize with appropriate arguments
                    if component_name == "language_memory_synthesis":
                        # Special case for synthesis integration
                        self.components[component_name] = component_class()
                    elif component_name == "v5_language_integration":
                        # Special case for V5 integration
                        if "language_memory_synthesis" in self.components:
                            self.components[component_name] = component_class(
                                language_memory_synthesis=self.components["language_memory_synthesis"]
                            )
                    else:
                        # Default initialization
                        self.components[component_name] = component_class()
                    
                    logger.info(f"✅ Initialized {component_name}")
                    self.component_status[component_name] = "active"
                    
                elif "function" in component_info:
                    # Function-based component
                    function_name = component_info["function"]
                    component_function = getattr(module, function_name)
                    
                    # Call the function to get the component
                    if component_name == "v5_bridge":
                        # Special case for V5 visualization bridge
                        bridge = component_function()
                        if bridge.is_visualization_available():
                            self.components[component_name] = bridge
                            logger.info(f"✅ Initialized {component_name}")
                            self.component_status[component_name] = "active"
                        else:
                            logger.warning(f"V5 visualization not available for {component_name}")
                            self.component_status[component_name] = "unavailable"
                    else:
                        # Default function call
                        result = component_function()
                        if isinstance(result, tuple) and len(result) >= 2 and result[0]:
                            self.components[component_name] = result[1]
                            logger.info(f"✅ Initialized {component_name}")
                            self.component_status[component_name] = "active"
                        else:
                            logger.warning(f"Function {function_name} did not return a valid component")
                            self.component_status[component_name] = "failed"
                
            except Exception as e:
                logger.error(f"Error initializing {component_name}: {str(e)}")
                self.component_status[component_name] = "error"
    
    def _initialize_integrations(self):
        """Initialize integration points between components"""
        logger.info("Initializing component integrations")
        
        # Connect Language Memory with Conversation Memory
        if "language_memory" in self.components and "conversation_memory" in self.components:
            logger.info("Connecting Language Memory with Conversation Memory")
            # Integration code remains the same
            
        # Connect LLM Enhancement with Language Memory Synthesis
        if "llm_enhancement" in self.components and "language_memory_synthesis" in self.components:
            logger.info("Connecting LLM Enhancement with Language Memory Synthesis")
            # Integration code remains the same
            
        # Connect V5 Bridge with Language Memory
        if "v5_bridge" in self.components and "language_memory" in self.components:
            logger.info("Connecting V5 Bridge with Language Memory")
            # Integration code remains the same
            
        # Connect Conscious Mirror Language with other components
        if "conscious_mirror_language" in self.components:
            logger.info("Initializing Conscious Mirror Language integrations")
            
            # Connect with Language Memory
            if "language_memory" in self.components:
                self.components["conscious_mirror_language"].language_memory = self.components["language_memory"]
                logger.info("Connected Conscious Mirror Language with Language Memory")
            
            # Connect with Node Consciousness (v7)
            if "v7_node_consciousness" in self.components:
                self.components["conscious_mirror_language"].node_consciousness = self.components["v7_node_consciousness"]
                logger.info("Connected Conscious Mirror Language with Node Consciousness (v7)")
            
            # Connect with Mirror Consciousness (v9)
            if "v9_mirror_consciousness" in self.components:
                self.components["conscious_mirror_language"].mirror_consciousness = self.components["v9_mirror_consciousness"]
                logger.info("Connected Conscious Mirror Language with Mirror Consciousness (v9)")
            
            # Connect with Language Memory Synthesis
            if "language_memory_synthesis" in self.components:
                # Register the conscious mirror as a component with the synthesis system
                try:
                    if hasattr(self.components["language_memory_synthesis"], "register_component"):
                        self.components["language_memory_synthesis"].register_component(
                            "conscious_mirror_language", 
                            self.components["conscious_mirror_language"]
                        )
                        logger.info("Registered Conscious Mirror Language with Language Memory Synthesis")
                except Exception as e:
                    logger.error(f"Error connecting Conscious Mirror Language with synthesis: {str(e)}")
                    
            # Initialize background consciousness processing
            if hasattr(self.components["conscious_mirror_language"], "_start_consciousness_thread"):
                logger.info("Starting background consciousness processing")
            
        # Connect Neural Linguistic Processor with other components
        if "neural_linguistic_processor" in self.components:
            logger.info("Initializing Neural Linguistic Processor integrations")
            
            # Connect with Language Memory
            if "language_memory" in self.components:
                self.components["neural_linguistic_processor"].language_memory = self.components["language_memory"]
                logger.info("Connected Neural Linguistic Processor with Language Memory")
            
            # Connect with Conscious Mirror Language
            if "conscious_mirror_language" in self.components:
                self.components["neural_linguistic_processor"].conscious_mirror_language = self.components["conscious_mirror_language"]
                logger.info("Connected Neural Linguistic Processor with Conscious Mirror Language")
            
            # Start background processing if both required components are available
            if "language_memory" in self.components and "conscious_mirror_language" in self.components:
                if hasattr(self.components["neural_linguistic_processor"], "_start_background_processing"):
                    logger.info("Starting Neural Linguistic Processor background processing")
        
        logger.info("Component integrations initialized")
    
    def _register_with_v10_system(self):
        """Register with v10 systems if available"""
        try:
            # Import v10 central node
            from src.v10.central_node import register_component, get_conscious_mirror
            
            # Register language components with v10 system
            if "language_memory" in self.components:
                register_component(self.components["language_memory"], name="language_memory")
                logger.info("Registered language_memory with v10 central node")
            
            if "language_memory_synthesis" in self.components:
                register_component(self.components["language_memory_synthesis"], name="language_memory_synthesis")
                logger.info("Registered language_memory_synthesis with v10 central node")
            
            if "conscious_mirror_language" in self.components:
                register_component(self.components["conscious_mirror_language"], name="conscious_mirror_language")
                logger.info("Registered conscious_mirror_language with v10 central node")
            
            if "neural_linguistic_processor" in self.components:
                register_component(self.components["neural_linguistic_processor"], name="neural_linguistic_processor")
                logger.info("Registered neural_linguistic_processor with v10 central node")
                
            logger.info("Registration with v10 system complete")
        except ImportError:
            logger.info("V10 central node not available for registration")
        except Exception as e:
            logger.error(f"Error registering with v10 system: {str(e)}")
    
    def synthesize_topic(self, topic: str, depth: int = 3) -> Dict[str, Any]:
        """
        Synthesize knowledge around a specific topic
        
        Args:
            topic: The topic to synthesize
            depth: Depth of search/synthesis (1-5)
            
        Returns:
            Synthesis results
        """
        logger.info(f"Synthesizing topic: {topic} with depth {depth}")
        self.metrics["operations_count"] += 1
        
        if "language_memory_synthesis" not in self.components:
            logger.error("Language memory synthesis component not available")
            return {"error": "Language memory synthesis component not available"}
        
        # Update component usage metrics
        self._update_metrics("language_memory_synthesis")
        
        # Perform synthesis
        try:
            synthesis = self.components["language_memory_synthesis"]
            results = synthesis.synthesize_topic(topic, depth)
            
            # If V5 visualization is available, also process with that
            if "v5_language_integration" in self.components:
                self.components["v5_language_integration"].process_language_data(topic, depth)
                self._update_metrics("v5_language_integration")
            
            return results
        except Exception as e:
            logger.error(f"Error synthesizing topic {topic}: {str(e)}")
            return {"error": str(e)}
    
    def store_memory(self, content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Store a new memory
        
        Args:
            content: Memory content
            metadata: Additional metadata
            
        Returns:
            Result of store operation
        """
        logger.info(f"Storing memory: {content[:50]}...")
        self.metrics["operations_count"] += 1
        
        if "conversation_memory" not in self.components:
            logger.error("Conversation memory component not available")
            return {"error": "Conversation memory component not available"}
        
        # Update component usage metrics
        self._update_metrics("conversation_memory")
        
        # Store memory
        try:
            memory = self.components["conversation_memory"]
            result = memory.store(content=content, metadata=metadata or {})
            return result
        except Exception as e:
            logger.error(f"Error storing memory: {str(e)}")
            return {"error": str(e)}
    
    def retrieve_memories(self, query: str, retrieval_type: str = "topic") -> List[Dict[str, Any]]:
        """
        Retrieve memories based on query
        
        Args:
            query: Search query
            retrieval_type: Type of retrieval (topic, keyword, text)
            
        Returns:
            List of matching memories
        """
        logger.info(f"Retrieving memories for {retrieval_type}: {query}")
        self.metrics["operations_count"] += 1
        
        if "conversation_memory" not in self.components:
            logger.error("Conversation memory component not available")
            return [{"error": "Conversation memory component not available"}]
        
        # Update component usage metrics
        self._update_metrics("conversation_memory")
        
        # Retrieve memories
        try:
            memory = self.components["conversation_memory"]
            
            if retrieval_type == "topic":
                results = memory.retrieve_by_topic(query)
            elif retrieval_type == "keyword":
                results = memory.retrieve_by_keyword(query)
            elif retrieval_type == "text":
                results = memory.search_text(query)
            else:
                results = []
                logger.error(f"Unknown retrieval type: {retrieval_type}")
            
            return results
        except Exception as e:
            logger.error(f"Error retrieving memories: {str(e)}")
            return [{"error": str(e)}]
    
    def enhance_llm_prompt(self, base_prompt: str, context: Dict[str, Any] = None) -> str:
        """
        Enhance an LLM prompt with memory context
        
        Args:
            base_prompt: Base prompt to enhance
            context: Additional context
            
        Returns:
            Enhanced prompt
        """
        logger.info(f"Enhancing LLM prompt: {base_prompt[:50]}...")
        self.metrics["operations_count"] += 1
        
        if "llm_enhancement" not in self.components:
            logger.error("LLM enhancement component not available")
            return base_prompt
        
        # Update component usage metrics
        self._update_metrics("llm_enhancement")
        
        # Enhance prompt
        try:
            enhancer = self.components["llm_enhancement"]
            enhanced_prompt = enhancer.enhance_prompt(base_prompt, context or {})
            return enhanced_prompt
        except Exception as e:
            logger.error(f"Error enhancing prompt: {str(e)}")
            return base_prompt
    
    def generate_training_data(self, topic: str, count: int = 5) -> List[str]:
        """
        Generate language training data
        
        Args:
            topic: Topic for training data
            count: Number of examples to generate
            
        Returns:
            List of training examples
        """
        logger.info(f"Generating training data for {topic}")
        self.metrics["operations_count"] += 1
        
        if "english_language_trainer" not in self.components:
            logger.error("English language trainer component not available")
            return [f"Training data for {topic} (trainer unavailable)"]
        
        # Update component usage metrics
        self._update_metrics("english_language_trainer")
        
        # Generate training data
        try:
            trainer = self.components["english_language_trainer"]
            training_data = trainer.generate_training_data(count=count, topic=topic)
            return training_data
        except Exception as e:
            logger.error(f"Error generating training data: {str(e)}")
            return [f"Error generating training data: {str(e)}"]
    
    def get_component(self, component_name: str) -> Optional[Any]:
        """
        Get a specific component by name
        
        Args:
            component_name: Name of component
            
        Returns:
            Component instance or None if not available
        """
        return self.components.get(component_name)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get status of the Central Language Node
        
        Returns:
            Status dictionary with component information and metrics
        """
        active_components = [name for name, status in self.component_status.items() 
                            if status == "active"]
        
        return {
            "active_components": active_components,
            "component_status": self.component_status,
            "metrics": self.metrics,
            "config": self.config
        }
    
    def _update_metrics(self, component_name: str):
        """Update component usage metrics"""
        if component_name not in self.metrics["component_usage"]:
            self.metrics["component_usage"][component_name] = 0
        self.metrics["component_usage"][component_name] += 1

    def process_with_consciousness(self, content: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process content with the V10 Conscious Mirror Language capabilities
        
        Args:
            content: Text content to process
            context: Optional context information
            
        Returns:
            Processing results with consciousness insights
        """
        if "conscious_mirror_language" not in self.components:
            # Fallback if consciousness component not available
            return {
                "text": content,
                "status": "consciousness_unavailable",
                "message": "Conscious Mirror Language component not available",
                "timestamp": datetime.now().isoformat()
            }
        
        # Default context if none provided
        context = context or {}
        
        # Add language memory synthesis context if available
        if "language_memory_synthesis" in self.components:
            try:
                synthesis_context = self.components["language_memory_synthesis"].generate_response_context(content)
                context["synthesis"] = synthesis_context
            except Exception as e:
                logger.error(f"Error generating synthesis context: {str(e)}")
        
        # Process with consciousness
        try:
            result = self.components["conscious_mirror_language"].process_with_consciousness(content, context)
            self._update_metrics("conscious_mirror_language")
            return result
        except Exception as e:
            logger.error(f"Error processing with consciousness: {str(e)}")
            return {
                "text": content,
                "status": "error",
                "message": f"Error processing with consciousness: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    def process_with_neural_linguistics(self, content: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process content with Neural Linguistic Processor
        
        Args:
            content: Text content to process
            context: Optional context information
            
        Returns:
            Processing results with neural linguistic insights
        """
        if "neural_linguistic_processor" not in self.components:
            # Fallback if processor not available
            return {
                "text": content,
                "status": "neural_processor_unavailable",
                "message": "Neural Linguistic Processor not available",
                "timestamp": datetime.now().isoformat()
            }
        
        # Default context if none provided
        context = context or {}
        
        # Add consciousness context if available
        if "conscious_mirror_language" in self.components:
            try:
                consciousness_results = self.components["conscious_mirror_language"].process_with_consciousness(content)
                context["consciousness"] = consciousness_results
            except Exception as e:
                logger.error(f"Error getting consciousness context: {str(e)}")
        
        # Process with neural linguistics
        try:
            result = self.components["neural_linguistic_processor"].process_text(content, context)
            self._update_metrics("neural_linguistic_processor")
            return result
        except Exception as e:
            logger.error(f"Error processing with neural linguistics: {str(e)}")
            return {
                "text": content,
                "status": "error",
                "message": f"Error processing with neural linguistics: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def get_semantic_network(self, word: str, depth: int = 2) -> Dict[str, Any]:
        """
        Get semantic network for a word
        
        Args:
            word: Center word for the semantic network
            depth: How deep to explore the network
            
        Returns:
            Semantic network data structure
        """
        if "neural_linguistic_processor" not in self.components:
            return {
                "status": "neural_processor_unavailable",
                "message": "Neural Linguistic Processor not available",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            network = self.components["neural_linguistic_processor"].get_semantic_network(word, depth)
            self._update_metrics("neural_linguistic_processor")
            return {
                "status": "success",
                "word": word,
                "depth": depth,
                "network": network,
                "node_count": len(network["nodes"]),
                "edge_count": len(network["edges"]),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting semantic network: {str(e)}")
            return {
                "status": "error",
                "message": f"Error getting semantic network: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def get_recursive_patterns(self, limit: int = 10) -> Dict[str, Any]:
        """
        Get recursive language patterns
        
        Args:
            limit: Maximum number of patterns to return
            
        Returns:
            Dictionary with recursive patterns
        """
        if "neural_linguistic_processor" not in self.components:
            return {
                "status": "neural_processor_unavailable",
                "message": "Neural Linguistic Processor not available",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            patterns = self.components["neural_linguistic_processor"].get_recursive_patterns(limit)
            self._update_metrics("neural_linguistic_processor")
            return {
                "status": "success",
                "patterns": patterns,
                "count": len(patterns),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting recursive patterns: {str(e)}")
            return {
                "status": "error",
                "message": f"Error getting recursive patterns: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def get_cross_domain_mappings(self, limit: int = 10) -> Dict[str, Any]:
        """
        Get cross-domain mappings between language and consciousness
        
        Args:
            limit: Maximum number of mappings to return
            
        Returns:
            Dictionary with cross-domain mappings
        """
        if "neural_linguistic_processor" not in self.components:
            return {
                "status": "neural_processor_unavailable",
                "message": "Neural Linguistic Processor not available",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            mappings = self.components["neural_linguistic_processor"].get_cross_domain_mappings(limit)
            self._update_metrics("neural_linguistic_processor")
            return {
                "status": "success",
                "mappings": mappings,
                "count": len(mappings),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting cross-domain mappings: {str(e)}")
            return {
                "status": "error",
                "message": f"Error getting cross-domain mappings: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }


def main(return_node=False):
    """
    Main function for testing the Central Language Node
    
    Args:
        return_node: If True, returns the node instance instead of running tests
    
    Returns:
        CentralLanguageNode instance if return_node is True, otherwise None
    """
    logger.info("Running Central Language Node tests")
    
    # Create node
    node = CentralLanguageNode()
    
    # Return node instance if requested
    if return_node:
        return node
    
    # Run some basic tests
    print("\n===== CENTRAL LANGUAGE NODE TEST =====\n")
    
    # Get status
    status = node.get_status()
    print(f"Central Language Node Status:")
    print(f"- Active components: {len(status['active_components'])}")
    print(f"- Total components: {len(status['component_status'])}")
    print(f"- Operations count: {status['metrics']['operations_count']}")
    
    # Test memory synthesis
    try:
        if "language_memory_synthesis" in node.components:
            print("\n=== Testing Memory Synthesis ===")
            result = node.synthesize_topic("neural networks", 2)
            print(f"Synthesized topic: neural networks")
            print(f"Result has {len(result.get('component_contributions', {}))} component contributions")
    except Exception as e:
        print(f"Synthesis test error: {str(e)}")
    
    # Test memory storage
    try:
        if "conversation_memory" in node.components:
            print("\n=== Testing Memory Storage ===")
            result = node.store_memory(
                "The Central Language Node integrates all language components.",
                {"test": True, "topic": "architecture"}
            )
            print(f"Memory stored with ID: {result.get('id', 'unknown')}")
    except Exception as e:
        print(f"Memory storage test error: {str(e)}")
    
    print("\n===== TEST COMPLETE =====\n")
    return None


if __name__ == "__main__":
    main() 