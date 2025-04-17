#!/usr/bin/env python
"""
V6 Symbolic State Manager

Manages the symbolic state of the V6 Portal of Contradiction system,
including breath states, glyph activations, and emotional resonance.
"""

import logging
import threading
import time
import random
from enum import Enum

try:
    from src.v5.ui.qt_compat import QtCore
    Signal = QtCore.Signal
except ImportError:
    try:
        from PySide6.QtCore import Signal
    except ImportError:
        # Mock Signal for testing
        class MockSignal:
            def __init__(self, *args):
                self.callbacks = []
            def connect(self, callback):
                self.callbacks.append(callback)
            def emit(self, *args):
                for callback in self.callbacks:
                    callback(*args)
        Signal = MockSignal

logger = logging.getLogger("SymbolicStateManager")

class BreathPhase(Enum):
    """Breath phases for breath integration"""
    INHALE = "inhale"
    HOLD = "hold"
    EXHALE = "exhale"
    REST = "rest"

class Element(Enum):
    """Elemental glyphs"""
    FIRE = "fire"
    WATER = "water"
    AIR = "air"
    EARTH = "earth"

class ProcessState(Enum):
    """Process states for symbolic visualization"""
    PROCESSING = "processing"
    REFLECTING = "reflecting"
    RECORDING = "recording"
    INTEGRATING = "integrating"

class EmotionalTone(Enum):
    """Emotional tones for resonance"""
    COMPASSION = "compassion"
    ANALYSIS = "analysis"
    CREATIVITY = "creativity"
    INSIGHT = "insight"

class V6SymbolicStateManager(QtCore.QObject):
    """
    Manages the symbolic state of the V6 Portal of Contradiction
    
    Coordinates breath states, glyph activations, and emotional resonance
    across the system, providing a unified symbolic presence layer.
    """
    
    # Define signals
    state_changed = Signal(dict)
    breath_phase_changed = Signal(str)
    element_changed = Signal(str)
    process_changed = Signal(str)
    emotion_changed = Signal(str)
    glyph_activated = Signal(str, float)  # glyph_id, strength
    contradiction_detected = Signal(float)  # intensity
    
    def __init__(self, socket_manager=None):
        super().__init__()
        
        # Socket manager for communication
        self.socket_manager = socket_manager
        
        # Current state
        self.symbolic_state = {
            "breath_phase": BreathPhase.INHALE.value,
            "active_element": Element.AIR.value,
            "active_process": ProcessState.REFLECTING.value,
            "active_emotion": EmotionalTone.INSIGHT.value,
            "glyph_activations": {},
            "contradiction_level": 0.0,
            "timestamp": self._get_timestamp()
        }
        
        # Breath cycle parameters
        self.breath_cycle_active = False
        self.breath_cycle_duration = 12.0  # seconds for full cycle
        self.breath_phase_durations = {
            BreathPhase.INHALE.value: 0.3,  # 30% of cycle
            BreathPhase.HOLD.value: 0.2,    # 20% of cycle
            BreathPhase.EXHALE.value: 0.4,  # 40% of cycle
            BreathPhase.REST.value: 0.1     # 10% of cycle
        }
        
        # Element-breath associations
        self.element_breath_map = {
            BreathPhase.INHALE.value: Element.FIRE.value,
            BreathPhase.HOLD.value: Element.EARTH.value,
            BreathPhase.EXHALE.value: Element.WATER.value,
            BreathPhase.REST.value: Element.AIR.value
        }
        
        # Initialize glyphs
        self.glyphs = {
            "fire": {"name": "Fire", "symbol": "🜂", "activation": 0.0},
            "water": {"name": "Water", "symbol": "🜄", "activation": 0.0},
            "air": {"name": "Air", "symbol": "🜁", "activation": 0.0},
            "earth": {"name": "Earth", "symbol": "🜃", "activation": 0.0},
            "processing": {"name": "Processing", "symbol": "⚙️", "activation": 0.0},
            "reflecting": {"name": "Reflecting", "symbol": "🔄", "activation": 0.0},
            "recording": {"name": "Recording", "symbol": "📝", "activation": 0.0},
            "integrating": {"name": "Integrating", "symbol": "🔗", "activation": 0.0},
            "compassion": {"name": "Compassion", "symbol": "❤️", "activation": 0.0},
            "analysis": {"name": "Analysis", "symbol": "🧠", "activation": 0.0},
            "creativity": {"name": "Creativity", "symbol": "✨", "activation": 0.0},
            "insight": {"name": "Insight", "symbol": "💡", "activation": 0.0}
        }
        
        # Register socket handlers
        if self.socket_manager:
            self.socket_manager.register_handler("breath_state", self.handle_breath_state)
            self.socket_manager.register_handler("glyph_activation", self.handle_glyph_activation)
            self.socket_manager.register_handler("emotional_state", self.handle_emotional_state)
            logger.info("Registered symbolic state handlers with socket manager")
        
        logger.info("Symbolic State Manager initialized")
    
    def start_breath_cycle(self):
        """Start automatic breath cycling"""
        if self.breath_cycle_active:
            return
            
        self.breath_cycle_active = True
        
        # Start breath cycle thread
        self.breath_thread = threading.Thread(
            target=self._run_breath_cycle,
            daemon=True,
            name="BreathCycleThread"
        )
        self.breath_thread.start()
        
        logger.info("Breath cycle started")
    
    def stop_breath_cycle(self):
        """Stop automatic breath cycling"""
        self.breath_cycle_active = False
        logger.info("Breath cycle stopped")
    
    def set_breath_phase(self, phase):
        """Set the current breath phase"""
        if phase not in [p.value for p in BreathPhase]:
            logger.warning(f"Invalid breath phase: {phase}")
            return False
            
        # Update state
        self.symbolic_state["breath_phase"] = phase
        
        # Update associated element
        if phase in self.element_breath_map:
            element = self.element_breath_map[phase]
            self.activate_element(element)
        
        # Emit signals
        self.breath_phase_changed.emit(phase)
        self._emit_state_update()
        
        logger.debug(f"Breath phase set to: {phase}")
        return True
    
    def activate_element(self, element):
        """Activate an elemental glyph"""
        if element not in [e.value for e in Element]:
            logger.warning(f"Invalid element: {element}")
            return False
            
        # Update state
        self.symbolic_state["active_element"] = element
        
        # Update glyph activation
        for e in [e.value for e in Element]:
            activation = 1.0 if e == element else 0.2
            self.glyphs[e]["activation"] = activation
            self.symbolic_state["glyph_activations"][e] = activation
        
        # Emit signals
        self.element_changed.emit(element)
        self.glyph_activated.emit(element, 1.0)
        self._emit_state_update()
        
        logger.debug(f"Element activated: {element}")
        return True
    
    def set_process_state(self, process):
        """Set the current process state"""
        if process not in [p.value for p in ProcessState]:
            logger.warning(f"Invalid process state: {process}")
            return False
            
        # Update state
        self.symbolic_state["active_process"] = process
        
        # Update glyph activation
        for p in [p.value for p in ProcessState]:
            activation = 1.0 if p == process else 0.2
            self.glyphs[p]["activation"] = activation
            self.symbolic_state["glyph_activations"][p] = activation
        
        # Emit signals
        self.process_changed.emit(process)
        self.glyph_activated.emit(process, 1.0)
        self._emit_state_update()
        
        logger.debug(f"Process state set to: {process}")
        return True
    
    def set_emotional_tone(self, emotion):
        """Set the current emotional tone"""
        if emotion not in [e.value for e in EmotionalTone]:
            logger.warning(f"Invalid emotional tone: {emotion}")
            return False
            
        # Update state
        self.symbolic_state["active_emotion"] = emotion
        
        # Update glyph activation
        for e in [e.value for e in EmotionalTone]:
            activation = 1.0 if e == emotion else 0.2
            self.glyphs[e]["activation"] = activation
            self.symbolic_state["glyph_activations"][e] = activation
        
        # Emit signals
        self.emotion_changed.emit(emotion)
        self.glyph_activated.emit(emotion, 1.0)
        self._emit_state_update()
        
        logger.debug(f"Emotional tone set to: {emotion}")
        return True
    
    def detect_contradiction(self, message_a, message_b, intensity=0.5):
        """Detect contradiction between messages"""
        # This would implement actual contradiction detection in production
        # For now, just simulate with random chance
        
        # Update state
        self.symbolic_state["contradiction_level"] = intensity
        
        # Emit signal
        self.contradiction_detected.emit(intensity)
        self._emit_state_update()
        
        logger.debug(f"Contradiction detected with intensity: {intensity}")
        return True
    
    def get_state(self):
        """Get the current symbolic state"""
        self.symbolic_state["timestamp"] = self._get_timestamp()
        return self.symbolic_state
    
    def get_active_element(self):
        """Get the currently active element"""
        return self.symbolic_state["active_element"]
    
    def get_active_process(self):
        """Get the currently active process state"""
        return self.symbolic_state["active_process"]
    
    def get_active_emotion(self):
        """Get the currently active emotional tone"""
        return self.symbolic_state["active_emotion"]
    
    def get_breath_phase(self):
        """Get the current breath phase"""
        return self.symbolic_state["breath_phase"]
    
    def get_glyph_info(self, glyph_id):
        """Get information about a specific glyph"""
        if glyph_id in self.glyphs:
            return self.glyphs[glyph_id]
        else:
            logger.warning(f"Invalid glyph ID: {glyph_id}")
            return None
    
    def handle_breath_state(self, data):
        """Handle breath state event from socket"""
        phase = data.get("phase")
        if phase:
            self.set_breath_phase(phase)
    
    def handle_glyph_activation(self, data):
        """Handle glyph activation event from socket"""
        glyph_id = data.get("glyph_id")
        strength = data.get("strength", 1.0)
        
        if glyph_id in self.glyphs:
            # Update glyph activation
            self.glyphs[glyph_id]["activation"] = strength
            self.symbolic_state["glyph_activations"][glyph_id] = strength
            
            # Emit signal
            self.glyph_activated.emit(glyph_id, strength)
            self._emit_state_update()
    
    def handle_emotional_state(self, data):
        """Handle emotional state event from socket"""
        emotion = data.get("emotion")
        if emotion:
            self.set_emotional_tone(emotion)
    
    def _run_breath_cycle(self):
        """Run automatic breath cycling"""
        logger.debug("Starting breath cycle thread")
        
        while self.breath_cycle_active:
            try:
                # Inhale phase
                self.set_breath_phase(BreathPhase.INHALE.value)
                time.sleep(self.breath_cycle_duration * self.breath_phase_durations[BreathPhase.INHALE.value])
                
                # Hold phase
                self.set_breath_phase(BreathPhase.HOLD.value)
                time.sleep(self.breath_cycle_duration * self.breath_phase_durations[BreathPhase.HOLD.value])
                
                # Exhale phase
                self.set_breath_phase(BreathPhase.EXHALE.value)
                time.sleep(self.breath_cycle_duration * self.breath_phase_durations[BreathPhase.EXHALE.value])
                
                # Rest phase
                self.set_breath_phase(BreathPhase.REST.value)
                time.sleep(self.breath_cycle_duration * self.breath_phase_durations[BreathPhase.REST.value])
                
            except Exception as e:
                logger.error(f"Error in breath cycle: {e}")
                time.sleep(1)
    
    def _emit_state_update(self):
        """Emit state update signal and notify socket manager"""
        # Update timestamp
        self.symbolic_state["timestamp"] = self._get_timestamp()
        
        # Emit signal
        self.state_changed.emit(self.symbolic_state)
        
        # Notify socket manager
        if self.socket_manager:
            self.socket_manager.emit("symbolic_state_update", self.symbolic_state)
    
    def _get_timestamp(self):
        """Get ISO8601 timestamp"""
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) 