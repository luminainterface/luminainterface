"""
Breath Contradiction Bridge

This module bridges V6's Portal of Contradiction with V7's Breath Detection System,
allowing for breath patterns to influence contradiction processing and vice versa.
"""

import os
import sys
import time
import logging
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable

# Set up logging
logger = logging.getLogger("v7.breath_contradiction_bridge")

# Add project root to path if needed
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import V6 and V7 components
try:
    from src.v6.contradiction_processor import ContradictionProcessor, Contradiction
    V6_AVAILABLE = True
except ImportError:
    logger.warning("V6 Contradiction Processor not found, using mock implementation")
    V6_AVAILABLE = False

try:
    from src.v7.breath_detection import BreathDetector
    V7_BREATH_AVAILABLE = True
except ImportError:
    logger.warning("V7 Breath Detector not found, using mock implementation")
    V7_BREATH_AVAILABLE = False

class BreathContradictionBridge:
    """
    Bridges the V6 Contradiction Portal with V7 Breath Detection System.
    
    This bridge enables:
    1. Breath patterns to influence contradiction processing
    2. Contradictions to influence breath pattern recognition
    3. Synchronized evolution of both systems based on their states
    
    This creates a feedback loop that enhances both systems:
    - Breath patterns provide emotional context for contradiction resolution
    - Contradictions influence emotional state reflected in breath
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the bridge"""
        # Default configuration
        self.config = {
            "mock_mode": False,
            "auto_sync": True,  # Automatically synchronize states
            "bridge_active": True,  # Bridge is active
            "sync_interval": 1.0,  # Seconds between sync operations
            "breath_weight": 0.6,  # Weight of breath influence on contradictions
            "contradiction_weight": 0.4,  # Weight of contradiction influence on breath
            "portal_breath_map": {
                # Maps portal intensity to breath pattern
                0.0: "relaxed",
                0.3: "focused",
                0.6: "creative",
                0.8: "stressed",
                1.0: "meditative"
            },
            "breath_portal_map": {
                # Maps breath pattern to portal influence
                "relaxed": 0.2,
                "focused": 0.5,
                "creative": 0.7,
                "stressed": 0.9, 
                "meditative": 0.3
            }
        }
        
        # Update with custom configuration
        if config:
            self.config.update(config)
        
        # Initialize components
        self.contradiction_processor = None
        self.breath_detector = None
        
        # Initialize V6 components if available
        if V6_AVAILABLE:
            try:
                self.contradiction_processor = ContradictionProcessor(
                    config={"v7_integration_enabled": True}
                )
                logger.info("V6 Contradiction Processor initialized")
            except Exception as e:
                logger.error(f"Error initializing V6 Contradiction Processor: {e}")
        
        # Initialize V7 components if available
        if V7_BREATH_AVAILABLE:
            try:
                self.breath_detector = BreathDetector()
                logger.info("V7 Breath Detector initialized")
            except Exception as e:
                logger.error(f"Error initializing V7 Breath Detector: {e}")
        
        # Bridge state
        self.active = False
        self.last_sync = 0
        self.sync_thread = None
        
        # Event handlers
        self.event_handlers = []
        
        # Metrics
        self.metrics = {
            "syncs_performed": 0,
            "contradictions_processed": 0,
            "breath_patterns_processed": 0
        }
        
        logger.info("Breath Contradiction Bridge initialized")
    
    def start(self):
        """Start the bridge"""
        if not self.active:
            self.active = True
            
            # Register handlers
            if self.contradiction_processor:
                self.contradiction_processor.register_contradiction_handler(
                    self._handle_contradiction
                )
                self.contradiction_processor.register_portal_state_handler(
                    self._handle_portal_state
                )
            
            if self.breath_detector:
                self.breath_detector.register_pattern_handler(
                    self._handle_breath_pattern
                )
            
            # Start sync thread if auto-sync is enabled
            if self.config["auto_sync"]:
                self.sync_thread = threading.Thread(
                    target=self._sync_loop,
                    daemon=True,
                    name="BreathContradictionSyncThread"
                )
                self.sync_thread.start()
            
            logger.info("Breath Contradiction Bridge started")
            return True
        
        return False
    
    def stop(self):
        """Stop the bridge"""
        if self.active:
            self.active = False
            logger.info("Breath Contradiction Bridge stopped")
            return True
        
        return False
    
    def register_event_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Register an event handler for bridge events"""
        if handler not in self.event_handlers:
            self.event_handlers.append(handler)
            return True
        
        return False
    
    def _notify_event_handlers(self, event: Dict[str, Any]):
        """Notify all registered event handlers"""
        for handler in self.event_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")
    
    def _handle_contradiction(self, contradiction: Contradiction):
        """Handle a contradiction from the V6 Contradiction Processor"""
        if not self.active or not self.breath_detector:
            return
        
        logger.debug(f"Handling contradiction: {contradiction.id}")
        
        # Map contradiction types to breath patterns
        pattern_map = {
            "logical": "focused",
            "temporal": "creative",
            "spatial": "meditative",
            "causal": "stressed",
            "linguistic": "relaxed"
        }
        
        # Get the target breath pattern for this contradiction type
        target_pattern = pattern_map.get(contradiction.type, "focused")
        
        # Influence the breath detector
        if hasattr(self.breath_detector, "suggest_pattern"):
            weight = self.config["contradiction_weight"]
            self.breath_detector.suggest_pattern(target_pattern, weight)
            
            logger.debug(f"Suggested breath pattern {target_pattern} with weight {weight}")
        
        # Update metrics
        self.metrics["contradictions_processed"] += 1
        
        # Notify event handlers
        self._notify_event_handlers({
            "type": "contradiction_processed",
            "contradiction_id": contradiction.id,
            "contradiction_type": contradiction.type,
            "suggested_pattern": target_pattern,
            "timestamp": time.time()
        })
    
    def _handle_portal_state(self, intensity: float):
        """Handle a portal state change from the V6 Contradiction Processor"""
        if not self.active or not self.breath_detector:
            return
        
        logger.debug(f"Handling portal state change: intensity = {intensity}")
        
        # Find the closest mapped breath pattern for this intensity
        portal_breath_map = self.config["portal_breath_map"]
        keys = sorted(portal_breath_map.keys())
        
        # Find the closest key
        closest_key = keys[0]
        for key in keys:
            if key <= intensity:
                closest_key = key
            else:
                break
        
        target_pattern = portal_breath_map[closest_key]
        
        # Influence the breath detector
        if hasattr(self.breath_detector, "suggest_pattern"):
            weight = self.config["contradiction_weight"]
            self.breath_detector.suggest_pattern(target_pattern, weight)
            
            logger.debug(f"Suggested breath pattern {target_pattern} from portal intensity {intensity}")
        
        # Notify event handlers
        self._notify_event_handlers({
            "type": "portal_state_changed",
            "intensity": intensity,
            "suggested_pattern": target_pattern,
            "timestamp": time.time()
        })
    
    def _handle_breath_pattern(self, pattern: str, confidence: float):
        """Handle a breath pattern from the V7 Breath Detector"""
        if not self.active or not self.contradiction_processor:
            return
        
        logger.debug(f"Handling breath pattern: {pattern} (confidence: {confidence})")
        
        # Get the portal influence for this breath pattern
        breath_portal_map = self.config["breath_portal_map"]
        portal_influence = breath_portal_map.get(pattern, 0.5)
        
        # Adjust by confidence
        adjusted_influence = portal_influence * confidence
        
        # Influence the portal state if supported
        if hasattr(self.contradiction_processor, "set_portal_intensity"):
            weight = self.config["breath_weight"]
            weighted_influence = adjusted_influence * weight
            
            self.contradiction_processor.set_portal_intensity(weighted_influence)
            
            logger.debug(f"Set portal intensity to {weighted_influence} from breath pattern {pattern}")
        
        # Update metrics
        self.metrics["breath_patterns_processed"] += 1
        
        # Notify event handlers
        self._notify_event_handlers({
            "type": "breath_pattern_processed",
            "pattern": pattern,
            "confidence": confidence,
            "portal_influence": adjusted_influence,
            "timestamp": time.time()
        })
    
    def _sync_states(self):
        """Synchronize states between systems"""
        if not self.active:
            return
        
        logger.debug("Synchronizing states between V6 and V7")
        
        try:
            # Get current states
            portal_state = None
            breath_state = None
            
            if self.contradiction_processor and hasattr(self.contradiction_processor, "get_portal_state"):
                portal_state = self.contradiction_processor.get_portal_state()
            
            if self.breath_detector and hasattr(self.breath_detector, "get_current_state"):
                breath_state = self.breath_detector.get_current_state()
            
            # No states to sync
            if not portal_state or not breath_state:
                return
            
            # Portal -> Breath influence
            if portal_state.get("active", False):
                intensity = portal_state.get("intensity", 0.0)
                
                # Find the closest mapped breath pattern for this intensity
                portal_breath_map = self.config["portal_breath_map"]
                keys = sorted(portal_breath_map.keys())
                
                # Find the closest key
                closest_key = keys[0]
                for key in keys:
                    if key <= intensity:
                        closest_key = key
                    else:
                        break
                
                target_pattern = portal_breath_map[closest_key]
                
                # Influence the breath detector
                if hasattr(self.breath_detector, "suggest_pattern"):
                    weight = self.config["contradiction_weight"]
                    self.breath_detector.suggest_pattern(target_pattern, weight)
            
            # Breath -> Portal influence
            current_pattern = breath_state.get("current_pattern", "relaxed")
            confidence = breath_state.get("confidence", 1.0)
            
            # Get the portal influence for this breath pattern
            breath_portal_map = self.config["breath_portal_map"]
            portal_influence = breath_portal_map.get(current_pattern, 0.5)
            
            # Adjust by confidence
            adjusted_influence = portal_influence * confidence
            
            # Influence the portal state if supported
            if hasattr(self.contradiction_processor, "set_portal_intensity"):
                weight = self.config["breath_weight"]
                weighted_influence = adjusted_influence * weight
                
                self.contradiction_processor.set_portal_intensity(weighted_influence)
            
            # Update metrics
            self.metrics["syncs_performed"] += 1
            
            # Notify event handlers
            self._notify_event_handlers({
                "type": "states_synchronized",
                "portal_state": portal_state,
                "breath_state": breath_state,
                "timestamp": time.time()
            })
            
            logger.debug("States synchronized successfully")
            
        except Exception as e:
            logger.error(f"Error synchronizing states: {e}")
    
    def _sync_loop(self):
        """Background thread for synchronizing states"""
        while self.active:
            try:
                current_time = time.time()
                
                # Check if it's time to sync
                if current_time - self.last_sync >= self.config["sync_interval"]:
                    self._sync_states()
                    self.last_sync = current_time
                
                # Sleep to prevent high CPU usage
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"Error in sync loop: {e}")
                time.sleep(1)  # Wait longer on error
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the bridge"""
        return {
            "active": self.active,
            "v6_available": V6_AVAILABLE,
            "v7_breath_available": V7_BREATH_AVAILABLE,
            "contradiction_processor_active": 
                self.contradiction_processor is not None and hasattr(self.contradiction_processor, "active") and self.contradiction_processor.active,
            "breath_detector_active": 
                self.breath_detector is not None and hasattr(self.breath_detector, "is_active") and self.breath_detector.is_active(),
            "last_sync": self.last_sync,
            "metrics": self.metrics,
            "config": {k: v for k, v in self.config.items() if k not in ["portal_breath_map", "breath_portal_map"]}
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get the metrics of the bridge"""
        return self.metrics
    
    def create_test_contradiction(self, type_override: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Create a test contradiction for testing the bridge"""
        if not self.contradiction_processor or not hasattr(self.contradiction_processor, "detect_contradiction"):
            return None
        
        # Create two contradictory statements
        statement1 = "The system always produces correct results"
        statement2 = "The system never produces correct results"
        
        # Detect the contradiction
        contradiction = self.contradiction_processor.detect_contradiction(statement1, statement2)
        
        if contradiction:
            logger.info(f"Created test contradiction: {contradiction.id}")
            return contradiction.to_dict()
        
        return None


# Testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create bridge
    bridge = BreathContradictionBridge({"mock_mode": True})
    
    # Register handler
    bridge.register_event_handler(
        lambda event: print(f"Event: {event['type']}")
    )
    
    # Start the bridge
    bridge.start()
    
    # Create a test contradiction
    test_contradiction = bridge.create_test_contradiction()
    if test_contradiction:
        print(f"Test contradiction created: {test_contradiction}")
    
    # Run for a while to see events
    print("Running for 30 seconds to observe bridge events...")
    
    try:
        for i in range(30):
            time.sleep(1)
            if i % 5 == 0:
                status = bridge.get_status()
                print(f"Bridge status: Active={status['active']}, Metrics={status['metrics']}")
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    # Stop the bridge
    bridge.stop()
    print("Bridge stopped") 