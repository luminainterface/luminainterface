import threading
import time
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum
from .base_node import BaseNode

class BreathPhase(Enum):
    """Breath cycle phases"""
    INHALE = "inhale"
    HOLD = "hold"
    EXHALE = "exhale"
    REST = "rest"

class BreathPattern(Enum):
    """Breath patterns"""
    RELAXED = "relaxed"      # Slow, deep breathing
    FOCUSED = "focused"      # Steady, controlled breathing
    RAPID = "rapid"         # Quick, shallow breathing
    MEDITATIVE = "meditative" # Very slow, deep breathing
    CREATIVE = "creative"    # Variable rhythm breathing

class BreathNode(BaseNode):
    """Node for managing breath cycles and patterns"""
    
    def __init__(self, node_id: str = None):
        super().__init__(node_id)
        
        # Breath cycle parameters
        self.breath_active = False
        self.current_phase = BreathPhase.REST
        self.current_pattern = BreathPattern.RELAXED
        self.breath_depth = 0.5  # Range 0.0 to 1.0
        
        # Timing parameters (in seconds)
        self.cycle_duration = 12.0
        self.phase_durations = {
            BreathPhase.INHALE: 0.3,  # 30% of cycle
            BreathPhase.HOLD: 0.2,    # 20% of cycle
            BreathPhase.EXHALE: 0.4,  # 40% of cycle
            BreathPhase.REST: 0.1     # 10% of cycle
        }
        
        # Pattern parameters
        self.pattern_settings = {
            BreathPattern.RELAXED: {
                "cycle_duration": 12.0,
                "depth": 0.5
            },
            BreathPattern.FOCUSED: {
                "cycle_duration": 8.0,
                "depth": 0.7
            },
            BreathPattern.RAPID: {
                "cycle_duration": 4.0,
                "depth": 0.3
            },
            BreathPattern.MEDITATIVE: {
                "cycle_duration": 16.0,
                "depth": 0.8
            },
            BreathPattern.CREATIVE: {
                "cycle_duration": 10.0,
                "depth": 0.6
            }
        }
        
        # Threading
        self.breath_thread = None
        self.breath_stop_event = threading.Event()
        
        # Update state
        self.state.update({
            "breath_active": self.breath_active,
            "current_phase": self.current_phase.value,
            "current_pattern": self.current_pattern.value,
            "breath_depth": self.breath_depth,
            "cycle_duration": self.cycle_duration
        })
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process breath-related data"""
        try:
            # Extract breath information from input
            pattern = data.get("pattern", "")
            depth = data.get("depth", None)
            
            # Update breath pattern if specified
            if pattern and pattern.lower() in [p.value for p in BreathPattern]:
                self.set_breath_pattern(BreathPattern(pattern.lower()))
            
            # Update breath depth if specified
            if depth is not None:
                self.breath_depth = max(0.0, min(1.0, float(depth)))
            
            # Update state
            self.state.update({
                "breath_active": self.breath_active,
                "current_phase": self.current_phase.value,
                "current_pattern": self.current_pattern.value,
                "breath_depth": self.breath_depth,
                "cycle_duration": self.cycle_duration
            })
            
            return {
                "status": "success",
                "breath_active": self.breath_active,
                "current_phase": self.current_phase.value,
                "current_pattern": self.current_pattern.value,
                "breath_depth": self.breath_depth,
                "cycle_duration": self.cycle_duration
            }
            
        except Exception as e:
            self.logger.error(f"Error processing breath data: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def activate(self) -> bool:
        """Activate the breath cycle"""
        if super().activate():
            return self.start_breath()
        return False
        
    def deactivate(self) -> bool:
        """Deactivate the breath cycle"""
        if super().deactivate():
            return self.stop_breath()
        return False
        
    def start_breath(self) -> bool:
        """Start the breath cycle"""
        if self.breath_active:
            return True
            
        try:
            self.breath_active = True
            self.breath_stop_event.clear()
            self.breath_thread = threading.Thread(
                target=self._run_breath_cycle,
                daemon=True,
                name="breath-cycle"
            )
            self.breath_thread.start()
            self.logger.info("Started breath cycle")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start breath cycle: {str(e)}")
            return False
            
    def stop_breath(self) -> bool:
        """Stop the breath cycle"""
        if not self.breath_active:
            return True
            
        try:
            self.breath_active = False
            self.breath_stop_event.set()
            if self.breath_thread:
                self.breath_thread.join(timeout=1.0)
                self.breath_thread = None
            self.logger.info("Stopped breath cycle")
            return True
        except Exception as e:
            self.logger.error(f"Failed to stop breath cycle: {str(e)}")
            return False
            
    def set_breath_pattern(self, pattern: BreathPattern):
        """Set the breath pattern"""
        if pattern not in BreathPattern:
            raise ValueError(f"Invalid breath pattern: {pattern}")
            
        self.current_pattern = pattern
        settings = self.pattern_settings[pattern]
        self.cycle_duration = settings["cycle_duration"]
        self.breath_depth = settings["depth"]
        self.logger.info(f"Set breath pattern to {pattern.value}")
        
    def _run_breath_cycle(self):
        """Run the breath cycle"""
        while self.breath_active and not self.breath_stop_event.is_set():
            try:
                # Calculate current phase
                cycle_time = time.time() % self.cycle_duration
                phase_time = 0.0
                
                # Determine current phase
                for phase in BreathPhase:
                    phase_duration = self.phase_durations[phase] * self.cycle_duration
                    if cycle_time < phase_time + phase_duration:
                        if phase != self.current_phase:
                            self.current_phase = phase
                            self.logger.debug(f"Breath phase: {phase.value}")
                        break
                    phase_time += phase_duration
                
                # Sleep briefly
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in breath cycle: {str(e)}")
                time.sleep(1.0)  # Sleep longer on error 