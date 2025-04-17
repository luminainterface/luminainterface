#!/usr/bin/env python3
"""
Central Language Node for Enhanced Language System

This class serves as the central coordinator for all language components,
providing a unified interface for language processing.
"""

import logging
import os
import json
from datetime import datetime
import threading
import time
import importlib
from typing import Dict, List, Any, Optional, Union

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/central_language_node.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CentralLanguageNode")

class CentralLanguageNode:
    """
    Central Language Node that integrates all language components with LLM weighing.
    
    This node coordinates:
    - Language Memory
    - Conscious Mirror Language
    - Neural Linguistic Processor
    - Recursive Pattern Analyzer
    - Neural Linguistic Flex Bridge (new)
    
    It provides a unified interface for language processing with consistent LLM weight
    application across all components.
    """
    
    def __init__(self, data_dir: str = "data", llm_weight: float = 0.5, nn_weight: float = 0.5):
        """
        Initialize the Central Language Node.
        
        Args:
            data_dir: Base directory for data storage
            llm_weight: Weight for LLM influence (0.0-1.0)
            nn_weight: Weight for neural network vs symbolic processing (0.0-1.0)
        """
        self.data_dir = data_dir
        self.llm_weight = llm_weight
        self.nn_weight = nn_weight  # New parameter for neural weight
        
        # Get the singleton LLM provider instance and set initial weights
        from .llm_provider import get_llm_provider
        self.llm_provider = get_llm_provider()
        self.llm_provider.set_llm_weight(llm_weight)
        self.llm_provider.set_nn_weight(nn_weight)
        
        # Component paths
        self.language_memory_dir = os.path.join(data_dir, "memory/language_memory")
        self.conscious_mirror_dir = os.path.join(data_dir, "v10")
        self.neural_processor_dir = os.path.join(data_dir, "neural_linguistic")
        self.recursive_patterns_dir = os.path.join(data_dir, "recursive_patterns")
        
        # Create necessary directories
        for directory in [
            self.language_memory_dir,
            self.conscious_mirror_dir,
            self.neural_processor_dir,
            self.recursive_patterns_dir
        ]:
            os.makedirs(directory, exist_ok=True)
        
        # Initialize components
        self.language_memory = None
        self.conscious_mirror = None
        self.neural_processor = None
        self.pattern_analyzer = None
        self.neural_flex_bridge = None  # New component
        
        # Database connection manager
        self.db_connection_manager = None
        
        # Cross-domain mappings
        self.cross_mappings = {
            "language_to_consciousness": {},
            "language_to_neural": {},
            "consciousness_to_neural": {},
            "recursive_to_language": {},
            "flex_to_language": {}  # New mapping for flex bridge
        }
        
        # Integration state
        self.integration_active = False
        self.integration_thread = None
        self.last_save_time = time.time()
        
        # Initialize database connection manager
        self._initialize_db_connection_manager()
        
        # Initialize components
        self._initialize_components()
        
        # Synchronize LLM weights after component initialization
        self._synchronize_llm_weights()
        
        # Connect all components to the database
        self._connect_components_to_database()
        
        # Start integration process
        self._start_integration_process()
        
        logger.info(f"Central Language Node initialized with LLM weight: {llm_weight}, NN weight: {nn_weight}")
    
    def _initialize_db_connection_manager(self):
        """Initialize the database connection manager"""
        try:
            # Import database connection manager
            from .database_connection_manager import get_database_connection_manager
            
            # Get the singleton instance
            self.db_connection_manager = get_database_connection_manager()
            
            # Initialize the connection manager
            if not self.db_connection_manager.initialized:
                self.db_connection_manager.initialize()
                
            logger.info("Database connection manager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database connection manager: {e}")
            self.db_connection_manager = None
    
    def _connect_components_to_database(self):
        """Connect all components to the database"""
        if not self.db_connection_manager:
            logger.warning("Cannot connect components to database: connection manager not initialized")
            return
            
        components = {
            "language_memory": self.language_memory,
            "conscious_mirror": self.conscious_mirror,
            "neural_processor": self.neural_processor,
            "pattern_analyzer": self.pattern_analyzer,
            "neural_flex_bridge": self.neural_flex_bridge,
            "central_language_node": self
        }
        
        # Register and connect each component
        for component_id, component in components.items():
            if component:
                try:
                    success = self.db_connection_manager.register_component(component_id, component)
                    if success:
                        logger.info(f"Component {component_id} registered with database connection manager")
                    else:
                        logger.warning(f"Failed to register component {component_id} with database connection manager")
                except Exception as e:
                    logger.error(f"Error registering component {component_id}: {e}")
        
        # Register periodic optimization hook
        try:
            self.db_connection_manager.register_component_hook(
                "central_language_node",
                "optimize_database",
                self._database_optimization_hook,
                interval=3600  # Run every hour
            )
            logger.info("Registered database optimization hook")
        except Exception as e:
            logger.error(f"Failed to register database optimization hook: {e}")
    
    def _database_optimization_hook(self):
        """Periodic database optimization hook"""
        try:
            logger.info("Running database optimization")
            
            # Get database connection manager status before optimization
            status_before = self.db_connection_manager.get_connection_status()
            
            # Run optimization
            results = self.db_connection_manager.optimize_all_connections()
            
            # Get status after optimization
            status_after = self.db_connection_manager.get_connection_status()
            
            logger.info(f"Database optimization completed: {results}")
            
            # Record optimization as a system metric if possible
            if hasattr(self, 'db_manager') and self.db_manager:
                self.db_manager.record_metric(
                    metric_name="database_optimization",
                    metric_value=1.0,
                    metric_type="performance",
                    details={
                        "results": results,
                        "status_before": status_before,
                        "status_after": status_after
                    }
                )
        except Exception as e:
            logger.error(f"Error in database optimization hook: {e}")
    
    def _initialize_components(self):
        """Initialize all language components."""
        try:
            # Import components using relative imports
            from .language_memory import LanguageMemory
            from .conscious_mirror_language import ConsciousMirrorLanguage
            from .neural_linguistic_processor import NeuralLinguisticProcessor
            from .recursive_pattern_analyzer import RecursivePatternAnalyzer
            
            # Import the new neural linguistic flex bridge
            try:
                from .neural_linguistic_flex_bridge import get_neural_linguistic_flex_bridge
                self.neural_flex_bridge = get_neural_linguistic_flex_bridge({
                    "mock_mode": False,  # Try to use real components
                    "embedding_dim": 256,
                    "learning_rate": 0.01,
                    "feedback_alpha": 0.3,
                    "pattern_weight": 0.7
                })
                logger.info("Neural Linguistic Flex Bridge initialized successfully")
                
                # Start the bridge if initialized successfully
                if self.neural_flex_bridge.initialized:
                    self.neural_flex_bridge.start()
                    logger.info("Neural Linguistic Flex Bridge started")
                else:
                    logger.warning("Neural Linguistic Flex Bridge initialization incomplete")
            except ImportError as e:
                logger.warning(f"Neural Linguistic Flex Bridge not available: {e}")
                self.neural_flex_bridge = None
            
            # Initialize components with both weights
            self.language_memory = LanguageMemory(
                data_dir=self.language_memory_dir,
                llm_weight=self.llm_weight,
                nn_weight=self.nn_weight
            )
            
            self.conscious_mirror = ConsciousMirrorLanguage(
                data_dir=self.conscious_mirror_dir,
                llm_weight=self.llm_weight,
                nn_weight=self.nn_weight
            )
            
            self.neural_processor = NeuralLinguisticProcessor(
                language_memory=self.language_memory,
                conscious_mirror_language=self.conscious_mirror
            )
            
            # Set the weights after initialization
            self.neural_processor.set_llm_weight(self.llm_weight)
            self.neural_processor.set_nn_weight(self.nn_weight)
            
            self.pattern_analyzer = RecursivePatternAnalyzer(
                data_dir=self.recursive_patterns_dir,
                llm_weight=self.llm_weight,
                nn_weight=self.nn_weight
            )
            
            logger.info("All components initialized successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import components: {e}")
            raise
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def _start_integration_process(self):
        """Start the background integration process."""
        self.integration_active = True
        self.integration_thread = threading.Thread(
            target=self._maintain_integration,
            daemon=True
        )
        self.integration_thread.start()
        logger.info("Integration process started")
    
    def _maintain_integration(self):
        """Maintain integration between components."""
        while self.integration_active:
            try:
                # Update cross-domain mappings
                self._update_cross_mappings()
                
                # Save state periodically (every 5 minutes)
                if time.time() - self.last_save_time > 300:
                    self.save_state()
                    self.last_save_time = time.time()
                
                # Sleep to prevent high CPU usage
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in integration process: {str(e)}")
                time.sleep(30)  # Longer sleep on error
    
    def _update_cross_mappings(self):
        """Update cross-domain mappings between components."""
        # This would implement the logic to update mappings between
        # different components' domains (language, consciousness, neural, recursive)
        # For now we'll keep it simple
        pass
    
    def _synchronize_llm_weights(self):
        """Ensure all components have the same LLM weight."""
        # Update the LLM provider first
        if hasattr(self.llm_provider, 'set_llm_weight'):
            self.llm_provider.set_llm_weight(self.llm_weight)
            self.llm_provider.set_nn_weight(self.nn_weight)
        
        # Update component weights
        if self.language_memory:
            self.language_memory.set_llm_weight(self.llm_weight)
            if hasattr(self.language_memory, 'set_nn_weight'):
                self.language_memory.set_nn_weight(self.nn_weight)
            logger.debug(f"Updated language memory weights: LLM={self.llm_weight}, NN={self.nn_weight}")
        
        if self.conscious_mirror:
            self.conscious_mirror.set_llm_weight(self.llm_weight)
            if hasattr(self.conscious_mirror, 'set_nn_weight'):
                self.conscious_mirror.set_nn_weight(self.nn_weight)
            logger.debug(f"Updated conscious mirror weights: LLM={self.llm_weight}, NN={self.nn_weight}")
        
        if self.neural_processor:
            self.neural_processor.set_llm_weight(self.llm_weight)
            if hasattr(self.neural_processor, 'set_nn_weight'):
                self.neural_processor.set_nn_weight(self.nn_weight)
            logger.debug(f"Updated neural processor weights: LLM={self.llm_weight}, NN={self.nn_weight}")
        
        if self.pattern_analyzer:
            self.pattern_analyzer.set_llm_weight(self.llm_weight)
            if hasattr(self.pattern_analyzer, 'set_nn_weight'):
                self.pattern_analyzer.set_nn_weight(self.nn_weight)
            logger.debug(f"Updated pattern analyzer weights: LLM={self.llm_weight}, NN={self.nn_weight}")
            
        # Update Neural Linguistic Flex Bridge if available
        if self.neural_flex_bridge:
            # The bridge might have different config parameters
            if hasattr(self.neural_flex_bridge, 'config'):
                self.neural_flex_bridge.config['llm_weight'] = self.llm_weight
                self.neural_flex_bridge.config['nn_weight'] = self.nn_weight
                
            # Update pattern weight based on nn_weight
            if hasattr(self.neural_flex_bridge, 'pattern_weight'):
                self.neural_flex_bridge.pattern_weight = self.nn_weight  # Higher NN weight = higher pattern influence
                
            logger.debug(f"Updated neural flex bridge weights: LLM={self.llm_weight}, NN={self.nn_weight}")
            
        logger.info(f"Synchronized weights across all components: LLM={self.llm_weight}, NN={self.nn_weight}")
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Process text through all language components.
        
        Args:
            text: Text to process
            
        Returns:
            Dictionary with results from all components and an integrated score
        """
        results = {}
        
        try:
            # Process through language memory
            if self.language_memory:
                # Store in memory
                sentence_id = self.language_memory.remember_sentence(text)
                
                # Get associations
                words = text.lower().split()
                associations = []
                
                # Get associations for substantive words
                for word in words:
                    if len(word) > 3:  # Only process words with length > 3
                        word_assocs = self.language_memory.recall_associations(word)
                        if word_assocs:
                            associations.extend(word_assocs)
                
                # Remove duplicates and sort by strength
                unique_assocs = {}
                for word, strength in associations:
                    if word not in unique_assocs or strength > unique_assocs[word]:
                        unique_assocs[word] = strength
                
                sorted_assocs = sorted(
                    [(word, strength) for word, strength in unique_assocs.items()],
                    key=lambda x: x[1],
                    reverse=True
                )
                
                results["memory_associations"] = sorted_assocs[:10]  # Top 10 associations
            
            # Process through conscious mirror language
            if self.conscious_mirror:
                consciousness_result = self.conscious_mirror.process_text(text)
                results.update(consciousness_result)
            
            # Process through neural linguistic processor
            if self.neural_processor:
                neural_result = self.neural_processor.process_text(text)
                results.update(neural_result)
            
            # Process through recursive pattern analyzer
            if self.pattern_analyzer:
                pattern_result = self.pattern_analyzer.analyze_text(text)
                results.update(pattern_result)
            
            # Calculate integrated score
            self._calculate_integrated_score(results)
            
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            results["error"] = str(e)
        
        return results
    
    def _calculate_integrated_score(self, results: Dict[str, Any]):
        """
        Calculate an integrated score based on component results.
        
        Args:
            results: Dictionary of component results to update with integrated score
        """
        # Get component scores with proper error handling
        try:
            # Extract primary metrics with defaults
            consciousness_level = float(results.get("consciousness_level", 0.0))
            neural_score = float(results.get("neural_linguistic_score", 0.0))
            
            # Handle different formats of confidence
            recursive_confidence = 0.0
            if "confidence" in results:
                recursive_confidence = float(results["confidence"])
            elif "pattern_confidence" in results:
                recursive_confidence = float(results["pattern_confidence"])
            
            # Apply neural network weight to balance between neural and symbolic processing
            neural_component = neural_score * self.nn_weight
            symbolic_component = consciousness_level * (1.0 - self.nn_weight)
            
            # Calculate a weighted combination with error prevention
            base_score = (neural_component * 0.6) + (symbolic_component * 0.4)
            
            # Add recursive confidence contribution if available
            if recursive_confidence > 0:
                base_score += (recursive_confidence * 0.2)
            
            # Apply LLM weight - higher weight means more LLM influence on the final score
            # Lower weight means more reliance on neural/consciousness components
            llm_factor = 1.0 + (self.llm_weight * 0.5)  # 1.0-1.5 range for enhancement factor
            final_score = base_score * llm_factor
            
            # Ensure score is within valid range
            final_score = max(0.0, min(1.0, final_score))
            
            # Update results
            results["final_score"] = round(final_score, 3)
            results["llm_weight_used"] = self.llm_weight
            results["nn_weight_used"] = self.nn_weight
            
        except Exception as e:
            # Handle errors gracefully
            logger.error(f"Error calculating integrated score: {e}")
            results["final_score"] = 0.5  # Default fallback score
            results["llm_weight_used"] = self.llm_weight
            results["nn_weight_used"] = self.nn_weight
            results["score_error"] = str(e)
    
    def process_consciousness_focused(self, text: str) -> Dict[str, Any]:
        """
        Process text with focus on consciousness aspects.
        
        Args:
            text: Text to process
            
        Returns:
            Dictionary with consciousness-focused results
        """
        if self.conscious_mirror:
            return self.conscious_mirror.process_text(text)
        return {"error": "Conscious Mirror Language component not available"}
    
    def process_neural_linguistic(self, text: str) -> Dict[str, Any]:
        """
        Process text with focus on neural linguistic aspects.
        
        Args:
            text: Text to process
            
        Returns:
            Dictionary with neural linguistic results
        """
        if self.neural_processor:
            return self.neural_processor.process_text(text)
        return {"error": "Neural Linguistic Processor component not available"}
    
    def get_semantic_network(self, seed_word: str, depth: int = 2) -> Dict[str, Any]:
        """
        Get a semantic network starting from a seed word.
        
        Args:
            seed_word: Starting word for the network
            depth: Depth of network to generate
            
        Returns:
            Dictionary with nodes and edges of the semantic network
        """
        if self.language_memory:
            return self.language_memory.get_semantic_network(seed_word, depth=depth)
        return {"nodes": [], "edges": []}
    
    def set_llm_weight(self, weight: float) -> bool:
        """
        Set the LLM weight for all components.
        
        Args:
            weight: New LLM weight (0.0-1.0)
            
        Returns:
            bool: Success or failure
        """
        if 0.0 <= weight <= 1.0:
            self.llm_weight = weight
            self._synchronize_llm_weights()
            logger.info(f"LLM weight updated to {weight} across all components")
            return True
        else:
            logger.warning(f"Invalid LLM weight: {weight}")
            return False
    
    def set_nn_weight(self, weight: float) -> bool:
        """
        Set the neural network weight for all components.
        
        Args:
            weight: New neural network weight (0.0-1.0)
            
        Returns:
            bool: Success or failure
        """
        if 0.0 <= weight <= 1.0:
            self.nn_weight = weight
            self._synchronize_llm_weights()  # This method handles both weights
            logger.info(f"Neural network weight updated to {weight} across all components")
            return True
        else:
            logger.warning(f"Invalid neural network weight: {weight}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get the current status of all system components."""
        status = {
            "llm_weight": self.llm_weight,
            "nn_weight": self.nn_weight,
            "components": {
                "language_memory": self.language_memory is not None,
                "conscious_mirror": self.conscious_mirror is not None,
                "neural_processor": self.neural_processor is not None,
                "pattern_analyzer": self.pattern_analyzer is not None,
                "neural_flex_bridge": self.neural_flex_bridge is not None  # New component status
            },
            "integration_active": self.integration_active,
            "timestamp": time.time()
        }
        
        # Add detailed component status
        if self.language_memory and hasattr(self.language_memory, 'get_status'):
            status["language_memory_status"] = self.language_memory.get_status()
            
        if self.conscious_mirror and hasattr(self.conscious_mirror, 'get_status'):
            status["conscious_mirror_status"] = self.conscious_mirror.get_status()
            
        if self.neural_processor and hasattr(self.neural_processor, 'get_status'):
            status["neural_processor_status"] = self.neural_processor.get_status()
            
        if self.pattern_analyzer and hasattr(self.pattern_analyzer, 'get_status'):
            status["pattern_analyzer_status"] = self.pattern_analyzer.get_status()
        
        # Add Neural Linguistic Flex Bridge status
        if self.neural_flex_bridge and hasattr(self.neural_flex_bridge, 'get_status'):
            status["neural_flex_bridge_status"] = self.neural_flex_bridge.get_status()
        
        # Add cross-mapping stats
        cross_mapping_stats = {
            mapping_type: len(mappings) 
            for mapping_type, mappings in self.cross_mappings.items()
        }
        status["cross_mapping_stats"] = cross_mapping_stats
        
        return status
    
    def save_state(self):
        """Save the state of all components."""
        try:
            # Save language memory
            if self.language_memory:
                self.language_memory.save_memories()
            
            # Save cross-domain mappings
            mappings_path = os.path.join(self.data_dir, "cross_mappings.json")
            with open(mappings_path, 'w') as f:
                json.dump(self.cross_mappings, f)
            
            logger.info("Central Language Node state saved")
            
        except Exception as e:
            logger.error(f"Error saving state: {str(e)}")
    
    def shutdown(self):
        """Shutdown the node and all components."""
        logger.info("Shutting down Central Language Node")
        
        # Stop integration process
        self.integration_active = False
        if self.integration_thread and self.integration_thread.is_alive():
            self.integration_thread.join(timeout=2.0)
        
        # Save state before shutdown
        self.save_state()
        
        # Shutdown Neural Linguistic Flex Bridge if available
        if self.neural_flex_bridge and hasattr(self.neural_flex_bridge, 'stop'):
            try:
                self.neural_flex_bridge.stop()
                logger.info("Neural Linguistic Flex Bridge stopped")
            except Exception as e:
                logger.error(f"Error stopping Neural Linguistic Flex Bridge: {e}")
        
        # Shutdown database connection manager
        if self.db_connection_manager:
            try:
                self.db_connection_manager.shutdown()
                logger.info("Database connection manager shutdown")
            except Exception as e:
                logger.error(f"Error shutting down database connection manager: {e}")
        
        logger.info("Central Language Node shutdown complete")
    
    def __del__(self):
        """Ensure proper shutdown on deletion."""
        self.shutdown()

    def process_neural_linguistic_flex(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process text specifically through the Neural Linguistic Flex Bridge.
        
        This method connects language processing with neural networks via the flex bridge,
        enabling adaptive pattern recognition and bidirectional communication.
        
        Args:
            text: Text to process
            context: Additional context information (optional)
            
        Returns:
            Processing results with both linguistic and neural components
        """
        if not text:
            return {"error": "No text provided"}
            
        if not self.neural_flex_bridge:
            logger.warning("Neural Linguistic Flex Bridge not available")
            return {"error": "Neural Linguistic Flex Bridge not available"}
            
        logger.info(f"Processing text through Neural Linguistic Flex Bridge: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        try:
            # Process through the bridge
            result = self.neural_flex_bridge.process_text(text, context)
            
            # Enhance the result with memory context if available
            if self.language_memory:
                memory_context = self.language_memory.search(text, limit=3)
                if memory_context:
                    result["memory_context"] = memory_context
            
            # Add consciousness context if available
            if self.conscious_mirror:
                consciousness_context = self.conscious_mirror.reflect_on_text(text)
                if consciousness_context:
                    result["consciousness_context"] = consciousness_context
            
            # Calculate integrated score
            linguistic_score = 0.0
            neural_score = 0.0
            
            if "linguistic_analysis" in result and "pattern_params" in result["linguistic_analysis"]:
                params = result["linguistic_analysis"]["pattern_params"]
                linguistic_score = (
                    params.get("resonance_factor", 0.0) * 0.3 +
                    params.get("semantic_density", 0.0) * 0.4 +
                    params.get("pattern_complexity", 0.0) * 0.3
                )
            
            if "neural_result" in result and isinstance(result["neural_result"], dict):
                neural_data = result["neural_result"]
                # Extract some score from neural result if available
                if "processed" in neural_data and neural_data["processed"]:
                    neural_score = 0.8  # Arbitrary score for successfully processed data
            
            # Apply weights to scores
            weighted_score = (linguistic_score * (1 - self.nn_weight) + 
                             neural_score * self.nn_weight)
            
            result["integrated_score"] = weighted_score
            result["weights_applied"] = {
                "llm_weight": self.llm_weight,
                "nn_weight": self.nn_weight
            }
            
            # Store in cross-mappings
            if "pattern_result" in result and result["pattern_result"]:
                # Create a simplified key from the text
                key = text[:50].replace(" ", "_").lower()
                self.cross_mappings["flex_to_language"][key] = {
                    "timestamp": time.time(),
                    "integrated_score": weighted_score,
                    "has_pattern": True
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing through Neural Linguistic Flex Bridge: {e}")
            return {
                "error": str(e),
                "text": text
            }


def test_central_language_node():
    """Test the Central Language Node functionality."""
    node = CentralLanguageNode(llm_weight=0.5, nn_weight=0.5)
    
    # Test processing
    test_text = "The neural network analyzes language patterns recursively."
    results = node.process_text(test_text)
    
    print(f"Processing results:")
    print(f"- Consciousness level: {results.get('consciousness_level', 0)}")
    print(f"- Neural linguistic score: {results.get('neural_linguistic_score', 0)}")
    print(f"- Self-references detected: {results.get('self_references', 0)}")
    print(f"- Integrated score: {results.get('final_score', 0)}")
    
    # Test LLM weight adjustment
    node.set_llm_weight(0.8)
    results_high_llm = node.process_text(test_text)
    print(f"\nResults with high LLM weight (0.8):")
    print(f"- Integrated score: {results_high_llm.get('final_score', 0)}")
    
    # Cleanup
    node.shutdown()


if __name__ == "__main__":
    test_central_language_node() 