"""
Dream Controller Module for V7 Dream Mode

This module implements the Dream Controller component of the Dream Mode system,
which coordinates the dream cycle and manages transitions between dream states.
"""

import logging
import random
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from enum import Enum, auto

# Set up logging
logger = logging.getLogger("lumina_v7.dream_controller")

class DreamState(Enum):
    """Enum representing possible states in the dream cycle"""
    INACTIVE = auto()
    INITIATION = auto()
    LIGHT_DREAM = auto()
    DEEP_DREAM = auto()
    INTEGRATION = auto()
    TRANSITION_OUT = auto()
    AWAKENING = auto()

class DreamController:
    """
    Coordinates the dream cycle and manages transitions between dream states
    
    Key features:
    - Dream state management (initiation, ongoing, awakening)
    - Coordination of memory consolidation and pattern synthesis
    - Dream cycle timing and intensity control
    - Dream record creation and archiving
    - Integration with other V7 components
    """
    
    def __init__(self, memory_consolidator=None, pattern_synthesizer=None, dream_archive=None, 
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Dream Controller
        
        Args:
            memory_consolidator: MemoryConsolidator instance (optional)
            pattern_synthesizer: PatternSynthesizer instance (optional)
            dream_archive: DreamArchive instance (optional)
            config: Configuration dictionary (optional)
        """
        # Default configuration
        self.config = {
            "auto_dream_enabled": True,     # Enable automatic dream cycles
            "dream_interval_hours": 8,      # Hours between automatic dream cycles
            "min_dream_duration": 10,       # Minimum dream duration in minutes
            "max_dream_duration": 30,       # Maximum dream duration in minutes
            "dream_phase_count": 3,         # Number of phases in a dream cycle
            "phase_intensity_curve": [0.4, 1.0, 0.7],  # Intensity by phase
            "memory_consolidation_weight": 0.6,  # Weight for memory consolidation
            "pattern_synthesis_weight": 0.4,     # Weight for pattern synthesis
            "auto_awakening": True,         # Automatically awaken from dreams
            "dream_state_check_interval": 5.0,  # Seconds between state checks
            "default_dream_intensity": 0.7,      # Default intensity for dreams
            "dream_jitter": 0.2,            # Random variation in dream timing
            "archive_dreams": True          # Save dream records to archive
        }
        
        # Update with custom config
        if config:
            self.config.update(config)
        
        # Normalize weights to sum to 1.0
        total_weight = (self.config["memory_consolidation_weight"] + 
                        self.config["pattern_synthesis_weight"])
        
        if total_weight > 0:
            self.config["memory_consolidation_weight"] /= total_weight
            self.config["pattern_synthesis_weight"] /= total_weight
        
        # Components
        self.memory_consolidator = memory_consolidator
        self.pattern_synthesizer = pattern_synthesizer
        self.dream_archive = dream_archive
        
        # Dream state
        self.dream_state = {
            "active": False,
            "phase": None,
            "intensity": 0.0,
            "start_time": None,
            "planned_end_time": None,
            "current_dream_id": None,
            "phase_index": 0,
            "metrics": {
                "memory_consolidation_count": 0,
                "pattern_synthesis_count": 0,
                "insights_generated": 0,
                "connections_created": 0
            }
        }
        
        # Current dream record
        self.current_dream = None
        
        # Dream history
        self.dream_history = []
        self.last_dream_time = None
        
        # Event callback
        self.dream_state_changed: Optional[Callable[[Dict[str, Any]], None]] = None
        
        # Threading
        self.dream_thread = None
        self.stop_event = threading.Event()
        
        # Automatic dream scheduling
        self.next_scheduled_dream = None
        if self.config["auto_dream_enabled"]:
            self._schedule_next_dream()
        
        logger.info("Dream Controller initialized")

    def start_dream(self, duration_minutes: Optional[int] = None, 
                   intensity: Optional[float] = None) -> Dict[str, Any]:
        """
        Start a new dream cycle
        
        Args:
            duration_minutes: Duration in minutes (None for default)
            intensity: Dream intensity from 0.0-1.0 (None for default)
            
        Returns:
            Dict with dream initiation status
        """
        # Check if already dreaming
        if self.dream_state["active"]:
            logger.warning("Cannot start dream - already dreaming")
            return {
                "success": False,
                "message": "Already in dream state",
                "dream_id": self.dream_state["current_dream_id"]
            }
        
        # Use defaults if not specified
        if duration_minutes is None:
            min_duration = self.config["min_dream_duration"]
            max_duration = self.config["max_dream_duration"]
            duration_minutes = int((min_duration + max_duration) / 2)
        
        if intensity is None:
            intensity = self.config["default_dream_intensity"]
            # Add some jitter to the intensity
            jitter = self.config["dream_jitter"]
            intensity += (random.random() * 2 - 1) * jitter
            intensity = max(0.1, min(1.0, intensity))
        
        # Generate dream ID
        dream_id = str(uuid.uuid4())
        
        # Create dream record
        self.current_dream = {
            "id": dream_id,
            "start_time": datetime.now().isoformat(),
            "planned_duration_minutes": duration_minutes,
            "intensity": intensity,
            "phases": [],
            "insights": [],
            "connections": [],
            "metrics": {
                "memory_consolidation_count": 0,
                "pattern_synthesis_count": 0,
                "insights_generated": 0,
                "connections_created": 0,
                "planned_duration_minutes": duration_minutes,
                "actual_duration_minutes": 0
            },
            "status": "initiated"
        }
        
        # Update dream state
        self.dream_state.update({
            "active": True,
            "phase": "initiation",
            "intensity": intensity,
            "start_time": datetime.now().isoformat(),
            "planned_end_time": (datetime.now() + timedelta(minutes=duration_minutes)).isoformat(),
            "current_dream_id": dream_id,
            "phase_index": 0
        })
        
        # Reset metrics
        self.dream_state["metrics"] = {
            "memory_consolidation_count": 0,
            "pattern_synthesis_count": 0,
            "insights_generated": 0,
            "connections_created": 0
        }
        
        # Trigger callback
        if self.dream_state_changed:
            self.dream_state_changed(self.dream_state.copy())
        
        # Start dream thread
        self.stop_event.clear()
        self.dream_thread = threading.Thread(
            target=self._dream_cycle,
            daemon=True,
            name="dream-controller"
        )
        self.dream_thread.start()
        
        logger.info(f"Dream started with ID {dream_id}, duration {duration_minutes}m, intensity {intensity:.2f}")
        
        return {
            "success": True,
            "message": "Dream initiated",
            "dream_id": dream_id,
            "duration_minutes": duration_minutes,
            "intensity": intensity
        }
    
    def end_dream(self, reason: str = "manual") -> Dict[str, Any]:
        """
        End the current dream cycle
        
        Args:
            reason: Reason for ending the dream
            
        Returns:
            Dict with dream termination status
        """
        # Check if not dreaming
        if not self.dream_state["active"]:
            logger.warning("Cannot end dream - not in dream state")
            return {
                "success": False,
                "message": "Not in dream state"
            }
        
        # Signal thread to stop
        self.stop_event.set()
        
        # Wait for thread to finish (with timeout)
        if self.dream_thread and self.dream_thread.is_alive():
            self.dream_thread.join(timeout=10)
        
        # Calculate actual duration
        if self.current_dream:
            start_time = datetime.fromisoformat(self.current_dream["start_time"])
            end_time = datetime.now()
            duration_minutes = (end_time - start_time).total_seconds() / 60
            
            # Update dream record
            self.current_dream["end_time"] = end_time.isoformat()
            self.current_dream["metrics"]["actual_duration_minutes"] = duration_minutes
            self.current_dream["status"] = "completed"
            self.current_dream["termination_reason"] = reason
            
            # Add to history
            self.dream_history.append(self.current_dream)
            self.last_dream_time = end_time
            
            # Archive dream if enabled
            if self.config["archive_dreams"] and self.dream_archive:
                try:
                    self.dream_archive.store_dream(self.current_dream)
                    logger.info(f"Dream {self.current_dream['id']} archived")
                except Exception as e:
                    logger.error(f"Error archiving dream: {e}")
        
        # Update dream state
        old_dream_id = self.dream_state["current_dream_id"]
        self.dream_state.update({
            "active": False,
            "phase": None,
            "intensity": 0.0,
            "start_time": None,
            "planned_end_time": None,
            "current_dream_id": None,
            "phase_index": 0
        })
        
        # Trigger callback
        if self.dream_state_changed:
            self.dream_state_changed(self.dream_state.copy())
        
        # Schedule next automatic dream if enabled
        if self.config["auto_dream_enabled"]:
            self._schedule_next_dream()
        
        logger.info(f"Dream {old_dream_id} ended: {reason}")
        
        return {
            "success": True,
            "message": f"Dream ended: {reason}",
            "dream_id": old_dream_id,
            "dream_record": self.current_dream
        }
    
    def get_dream_state(self) -> Dict[str, Any]:
        """
        Get the current dream state
        
        Returns:
            Dict with current dream state
        """
        return self.dream_state.copy()
    
    def get_current_dream(self) -> Optional[Dict[str, Any]]:
        """
        Get the current dream record
        
        Returns:
            Dict with current dream or None if not dreaming
        """
        return self.current_dream.copy() if self.current_dream else None
    
    def get_dream_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent dream history
        
        Args:
            limit: Maximum number of dreams to return
            
        Returns:
            List of recent dream records
        """
        return self.dream_history[-limit:] if self.dream_history else []
    
    def _dream_cycle(self):
        """Main dream cycle processing loop"""
        try:
            # Get dream parameters
            dream_id = self.dream_state["current_dream_id"]
            start_time = datetime.fromisoformat(self.dream_state["start_time"])
            planned_end_time = datetime.fromisoformat(self.dream_state["planned_end_time"])
            base_intensity = self.dream_state["intensity"]
            
            # Calculate phase durations
            total_duration = (planned_end_time - start_time).total_seconds()
            phase_count = self.config["dream_phase_count"]
            phase_duration = total_duration / phase_count
            
            # Log dream start
            logger.info(f"Dream cycle starting for dream {dream_id}")
            
            # Record initiation phase
            self._record_dream_phase("initiation", base_intensity * 0.5)
            
            # Process each phase
            for phase_idx in range(phase_count):
                # Check if we should stop
                if self.stop_event.is_set():
                    logger.info(f"Dream cycle interrupted during phase {phase_idx}")
                    break
                
                # Calculate phase intensity based on curve
                intensity_curve = self.config["phase_intensity_curve"]
                if phase_idx < len(intensity_curve):
                    phase_intensity = base_intensity * intensity_curve[phase_idx]
                else:
                    # Default to base intensity if curve not defined for this phase
                    phase_intensity = base_intensity
                
                # Update phase state
                self.dream_state["phase"] = f"dream_phase_{phase_idx+1}"
                self.dream_state["phase_index"] = phase_idx + 1
                self.dream_state["intensity"] = phase_intensity
                
                # Record phase start
                self._record_dream_phase(f"phase_{phase_idx+1}", phase_intensity)
                
                # Trigger callback
                if self.dream_state_changed:
                    self.dream_state_changed(self.dream_state.copy())
                
                # Process memory consolidation
                if self.memory_consolidator:
                    try:
                        # Calculate time allocation based on weights
                        consolidation_time = phase_duration * self.config["memory_consolidation_weight"]
                        
                        # Run consolidation
                        consolidation_results = self.memory_consolidator.consolidate_memories(
                            intensity=phase_intensity,
                            time_limit=consolidation_time
                        )
                        
                        # Update metrics
                        if isinstance(consolidation_results, dict):
                            self.dream_state["metrics"]["memory_consolidation_count"] += consolidation_results.get("memories_consolidated", 0)
                            self.dream_state["metrics"]["connections_created"] += consolidation_results.get("connections_created", 0)
                            
                            # Update dream record
                            if self.current_dream:
                                self.current_dream["metrics"]["memory_consolidation_count"] += consolidation_results.get("memories_consolidated", 0)
                                self.current_dream["metrics"]["connections_created"] += consolidation_results.get("connections_created", 0)
                                
                                # Add connections to dream record
                                if "connections" in consolidation_results:
                                    self.current_dream["connections"].extend(consolidation_results["connections"])
                    
                    except Exception as e:
                        logger.error(f"Error during memory consolidation: {e}")
                
                # Process pattern synthesis
                if self.pattern_synthesizer:
                    try:
                        # Calculate time allocation based on weights
                        synthesis_time = phase_duration * self.config["pattern_synthesis_weight"]
                        
                        # Run synthesis
                        synthesis_results = self.pattern_synthesizer.synthesize_patterns(
                            intensity=phase_intensity,
                            time_limit=synthesis_time
                        )
                        
                        # Update metrics
                        if isinstance(synthesis_results, dict):
                            self.dream_state["metrics"]["pattern_synthesis_count"] += synthesis_results.get("new_patterns", 0)
                            self.dream_state["metrics"]["insights_generated"] += len(synthesis_results.get("insights", []))
                            
                            # Update dream record
                            if self.current_dream:
                                self.current_dream["metrics"]["pattern_synthesis_count"] += synthesis_results.get("new_patterns", 0)
                                self.current_dream["metrics"]["insights_generated"] += len(synthesis_results.get("insights", []))
                                
                                # Add insights to dream record
                                if "insights" in synthesis_results:
                                    self.current_dream["insights"].extend(synthesis_results["insights"])
                    
                    except Exception as e:
                        logger.error(f"Error during pattern synthesis: {e}")
                
                # Check if we should continue to next phase
                now = datetime.now()
                phase_end_time = start_time + timedelta(seconds=phase_duration * (phase_idx + 1))
                
                # Sleep until phase end time if needed
                if now < phase_end_time and not self.stop_event.is_set():
                    sleep_time = (phase_end_time - now).total_seconds()
                    # Sleep in small increments to check for stop event
                    while sleep_time > 0 and not self.stop_event.is_set():
                        time.sleep(min(1.0, sleep_time))
                        sleep_time -= 1.0
                
                # Check if we should stop (reached planned end time)
                if now >= planned_end_time and self.config["auto_awakening"]:
                    logger.info(f"Dream cycle reached planned end time after phase {phase_idx+1}")
                    break
            
            # Record awakening phase if not already stopped
            if not self.stop_event.is_set():
                self.dream_state["phase"] = "awakening"
                self.dream_state["intensity"] = base_intensity * 0.3
                
                # Trigger callback
                if self.dream_state_changed:
                    self.dream_state_changed(self.dream_state.copy())
                
                # Record awakening phase
                self._record_dream_phase("awakening", base_intensity * 0.3)
                
                # Sleep briefly during awakening
                time.sleep(min(5, phase_duration * 0.1))
                
                # End dream
                self.end_dream(reason="completed")
            
        except Exception as e:
            logger.error(f"Error during dream cycle: {e}")
            # Try to end dream gracefully on error
            try:
                self.end_dream(reason="error")
            except:
                # Last resort - reset dream state directly
                self.dream_state["active"] = False
                self.dream_state["phase"] = None
    
    def _record_dream_phase(self, phase_name: str, intensity: float):
        """
        Record a dream phase in the current dream record
        
        Args:
            phase_name: Name of the phase
            intensity: Intensity of the phase
        """
        if not self.current_dream:
            return
        
        phase_record = {
            "phase": phase_name,
            "time": datetime.now().isoformat(),
            "intensity": intensity
        }
        
        self.current_dream["phases"].append(phase_record)
        logger.debug(f"Recorded dream phase: {phase_name} with intensity {intensity:.2f}")
    
    def _schedule_next_dream(self):
        """Schedule the next automatic dream"""
        # Calculate next dream time
        now = datetime.now()
        
        if self.last_dream_time:
            # Base on last dream time
            hours = self.config["dream_interval_hours"]
            # Add some jitter
            jitter = self.config["dream_jitter"] * hours
            adjusted_hours = hours + (random.random() * 2 - 1) * jitter
            next_time = self.last_dream_time + timedelta(hours=adjusted_hours)
            
            # Ensure it's in the future
            if next_time <= now:
                next_time = now + timedelta(hours=1)
        else:
            # No previous dream, schedule in the near future
            next_time = now + timedelta(hours=1)
        
        self.next_scheduled_dream = next_time
        logger.info(f"Next dream scheduled for {next_time.isoformat()}")
    
    def check_scheduled_dreams(self) -> bool:
        """
        Check if it's time for a scheduled dream
        
        Returns:
            True if a dream was started, False otherwise
        """
        if not self.config["auto_dream_enabled"] or not self.next_scheduled_dream:
            return False
        
        # Check if it's time for the next dream
        if datetime.now() >= self.next_scheduled_dream and not self.dream_state["active"]:
            logger.info("Starting scheduled dream")
            self.start_dream()
            return True
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get controller statistics
        
        Returns:
            Dict with controller statistics
        """
        return {
            "total_dreams": len(self.dream_history),
            "average_duration": sum(d.get("metrics", {}).get("actual_duration_minutes", 0) 
                                  for d in self.dream_history) / max(1, len(self.dream_history)),
            "total_insights": sum(d.get("metrics", {}).get("insights_generated", 0) 
                               for d in self.dream_history),
            "total_connections": sum(d.get("metrics", {}).get("connections_created", 0) 
                                  for d in self.dream_history),
            "next_scheduled_dream": self.next_scheduled_dream.isoformat() if self.next_scheduled_dream else None,
            "auto_dream_enabled": self.config["auto_dream_enabled"],
            "current_state": self.dream_state
        }

    def get_visualization_data(self) -> Dict[str, Any]:
        """
        Get data for dream cycle visualization
        
        Returns:
            Dict with visualization-friendly data
        """
        # Basic state information
        vis_data = {
            "active": self.dream_state["active"],
            "phase": self.dream_state.get("phase"),
            "intensity": self.dream_state.get("intensity", 0.0),
            "metrics": self.dream_state.get("metrics", {}),
            "phase_index": self.dream_state.get("phase_index", 0)
        }
        
        # Add dream details if available
        if self.current_dream:
            # Extract key information for visualization
            insights = self.current_dream.get("insights", [])
            phases = self.current_dream.get("phases", [])
            
            # Process insight data for visualization
            processed_insights = []
            for insight in insights:
                # Extract the text and other key fields
                insight_data = {
                    "text": insight.get("text", ""),
                    "confidence": insight.get("confidence", 0.0),
                    "timestamp": insight.get("timestamp", ""),
                    "type": insight.get("type", "general")
                }
                processed_insights.append(insight_data)
            
            # Process phase data for visualization
            processed_phases = []
            for phase in phases:
                # Extract phase information
                phase_data = {
                    "phase": phase.get("phase", ""),
                    "time": phase.get("time", ""),
                    "intensity": phase.get("intensity", 0.0),
                    "duration_seconds": 0  # Will be calculated below
                }
                processed_phases.append(phase_data)
            
            # Calculate phase durations
            if len(processed_phases) > 1:
                for i in range(1, len(processed_phases)):
                    try:
                        prev_time = datetime.fromisoformat(processed_phases[i-1]["time"])
                        curr_time = datetime.fromisoformat(processed_phases[i]["time"])
                        duration = (curr_time - prev_time).total_seconds()
                        processed_phases[i-1]["duration_seconds"] = duration
                    except (ValueError, TypeError):
                        pass
            
            # Add to visualization data
            vis_data["dream_data"] = {
                "id": self.current_dream.get("id"),
                "start_time": self.current_dream.get("start_time"),
                "planned_duration": self.current_dream.get("planned_duration_minutes"),
                "insights": processed_insights,
                "insight_count": len(insights),
                "phases": processed_phases,
                "phase_count": len(phases),
                "connection_count": len(self.current_dream.get("connections", []))
            }
            
            # Add metrics for visualization
            if "metrics" in self.current_dream:
                vis_data["dream_metrics"] = self.current_dream["metrics"]
        
        # Add dream history summary
        if self.dream_history:
            # Get summary of recent dreams
            recent_dreams = self.dream_history[-10:]  # Last 10 dreams
            
            # Create summary data
            history_summary = []
            for dream in recent_dreams:
                summary = {
                    "id": dream.get("id"),
                    "start_time": dream.get("start_time"),
                    "duration": dream.get("metrics", {}).get("actual_duration_minutes", 0),
                    "insight_count": len(dream.get("insights", [])),
                    "connection_count": len(dream.get("connections", [])),
                }
                history_summary.append(summary)
            
            vis_data["dream_history"] = history_summary
        
        # Add scheduling information
        vis_data["next_scheduled_dream"] = (
            self.next_scheduled_dream.isoformat() if self.next_scheduled_dream else None
        )
        vis_data["auto_dream_enabled"] = self.config["auto_dream_enabled"]
        
        return vis_data


def get_dream_controller(memory_consolidator=None, pattern_synthesizer=None, 
                        dream_archive=None, config=None):
    """
    Factory function to create a DreamController instance
    
    Args:
        memory_consolidator: MemoryConsolidator instance (optional)
        pattern_synthesizer: PatternSynthesizer instance (optional)
        dream_archive: DreamArchive instance (optional)
        config: Configuration dictionary (optional)
        
    Returns:
        DreamController instance
    """
    return DreamController(
        memory_consolidator=memory_consolidator,
        pattern_synthesizer=pattern_synthesizer,
        dream_archive=dream_archive,
        config=config
    ) 