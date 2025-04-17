import time
import random
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
from threading import Lock
import math

# Configure logging
logger = logging.getLogger(__name__)

class SeedEngine:
    """
    Core engine for seeding, quantum infecting, and growing ideas through the system.
    Integrates with memory, glyph states, and echo spirals to create emergent patterns.
    """
    
    def __init__(self, central_node=None):
        self.central_node = central_node
        self.board: Dict[str, Dict] = {}  # Evidence board: seed phrases → nodes
        self.growth_lock = Lock()
        self.active_growths: Set[str] = set()
        self.quantum_infection = None
        self.echo_spiral = None
        self.glyph_state = None
        
        # Growth parameters
        self.growth_threshold = math.pi  # Critical mass for growth spurting
        self.resonance_decay = 0.95  # How quickly resonance fades
        self.max_threads = 100  # Maximum threads per seed
        
        # Initialize connections
        self._initialize_connections()
        
    def _initialize_connections(self):
        """Initialize connections to other system components"""
        if self.central_node:
            try:
                self.quantum_infection = self.central_node.get_processor('QuantumInfection')
                self.echo_spiral = self.central_node.get_node('EchoSpiralNode')
                self.glyph_state = self.central_node.get_node('GlyphStateNode')
            except Exception as e:
                logger.error(f"Error initializing connections: {str(e)}")
                
    def plant(self, seed_phrase: str, context: Optional[Dict] = None) -> Dict:
        """
        Begin the infection process with a seed phrase.
        
        Args:
            seed_phrase: The phrase or concept to plant
            context: Optional context for the seed
            
        Returns:
            Dictionary containing the planted seed's information
        """
        with self.growth_lock:
            if seed_phrase in self.board:
                logger.info(f"Seed '{seed_phrase}' already exists, updating...")
                return self._update_seed(seed_phrase, context)
                
            # Create new seed node
            node = {
                "origin": seed_phrase,
                "timestamp": datetime.now(),
                "glyph": self._suggest_glyph(seed_phrase),
                "threads": [],
                "emergence_score": 1.0,  # Initial score
                "status": "germinating",
                "context": context or {},
                "resonance_history": [],
                "growth_cycles": 0
            }
            
            self.board[seed_phrase] = node
            logger.info(f"Planted new seed: '{seed_phrase}'")
            
            # Start quantum infection without initial decay
            self._propagate(seed_phrase, node, apply_decay=False)
            
            return node
            
    def _update_seed(self, seed_phrase: str, context: Optional[Dict] = None) -> Dict:
        """Update an existing seed with new context"""
        node = self.board[seed_phrase]
        if context:
            node["context"].update(context)
        node["timestamp"] = datetime.now()
        self._propagate(seed_phrase, node)
        return node
        
    def _propagate(self, phrase: str, node: Dict, apply_decay: bool = True):
        """Quantum-infect system: cross-link seed with memory, glyphs, echoes"""
        try:
            # Get recent memories
            if self.echo_spiral:
                recent_memories = self.echo_spiral.get_recent_memories()
                new_threads = []
                
                for memory in recent_memories:
                    if self._check_resonance(phrase, memory, node["context"]):
                        thread = {
                            "resonance": random.uniform(0.5, 1.0),
                            "matched": memory["content"],
                            "echo": self.echo_spiral.find_related(memory["content"]),
                            "timestamp": datetime.now()
                        }
                        new_threads.append(thread)
                        
                # Add new threads up to max limit
                space_available = self.max_threads - len(node["threads"])
                if space_available > 0:
                    node["threads"].extend(new_threads[:space_available])
                    for thread in new_threads[:space_available]:
                        node["emergence_score"] += thread["resonance"]
                        
            # Check for growth spurt
            if node["emergence_score"] > self.growth_threshold:
                self._trigger_growth_spurt(node)
                
            # Update resonance history
            node["resonance_history"].append({
                "timestamp": datetime.now(),
                "score": node["emergence_score"]
            })
            
            # Apply resonance decay if needed
            if apply_decay:
                node["emergence_score"] *= self.resonance_decay
                
        except Exception as e:
            logger.error(f"Error propagating seed '{phrase}': {str(e)}")
            
    def _check_resonance(self, phrase: str, memory: Dict, context: Dict) -> bool:
        """Check if memory resonates with seed phrase and context"""
        content = memory.get("content", "").lower()
        phrase = phrase.lower()
        
        # Direct match
        if phrase in content:
            return True
            
        # Context match - check if any context value appears in content
        if context:
            content_words = set(content.split())
            for value in context.values():
                value_words = set(str(value).lower().split())
                if value_words & content_words:  # Check for any word overlap
                    return True
                    
        return False
        
    def _trigger_growth_spurt(self, node: Dict):
        """Trigger visual and system effects from growth spurt"""
        try:
            node["status"] = "growthspurting"
            node["growth_cycles"] += 1
            
            # Activate glyph
            if self.glyph_state:
                self.glyph_state.activate(node["glyph"])
                
            # Insert echo
            if self.echo_spiral and hasattr(self.echo_spiral, 'insert_echo'):
                self.echo_spiral.insert_echo(
                    f"Seed '{node['origin']}' sprouted under glyph {node['glyph']}"
                )
                
            # Quantum infect
            if self.quantum_infection:
                self.quantum_infection.infect_data({
                    "type": "growth_spurt",
                    "seed": node["origin"],
                    "glyph": node["glyph"],
                    "score": node["emergence_score"]
                })
                
            logger.info(f"Growth spurt triggered for seed: '{node['origin']}'")
            
        except Exception as e:
            logger.error(f"Error triggering growth spurt: {str(e)}")
            
    def _suggest_glyph(self, text: str) -> str:
        """Suggest appropriate glyph based on text content"""
        text = text.lower()
        
        # Elemental glyphs
        if any(word in text for word in ["fire", "rage", "passion", "anger"]):
            return "🜂"  # Fire
        elif any(word in text for word in ["water", "grief", "flow", "emotion"]):
            return "🜄"  # Water
        elif any(word in text for word in ["air", "thought", "mind", "breath"]):
            return "🜁"  # Air
        elif any(word in text for word in ["earth", "silence", "ground", "still"]):
            return "🜃"  # Earth
            
        # Alchemical glyphs
        alchemical_glyphs = ["🜔", "🝊", "🜚", "🜕", "🜖", "🜗", "🜘", "🜙"]
        return random.choice(alchemical_glyphs)
        
    def get_evidence_board(self) -> Dict:
        """Return a snapshot of current threads for UI display"""
        return {
            phrase: {
                "status": node["status"],
                "score": round(node["emergence_score"], 2),
                "glyph": node["glyph"],
                "thread_count": len(node["threads"]),
                "growth_cycles": node["growth_cycles"],
                "age": (datetime.now() - node["timestamp"]).total_seconds()
            }
            for phrase, node in self.board.items()
        }
        
    def get_seed_details(self, seed_phrase: str) -> Optional[Dict]:
        """Get detailed information about a specific seed"""
        return self.board.get(seed_phrase)
        
    def prune_inactive_seeds(self, max_age_seconds: int = 86400):
        """Remove seeds that haven't shown activity in specified time"""
        current_time = datetime.now()
        to_remove = []
        
        for phrase, node in self.board.items():
            age = (current_time - node["timestamp"]).total_seconds()
            if age > max_age_seconds and node["status"] != "growthspurting":
                to_remove.append(phrase)
                
        for phrase in to_remove:
            del self.board[phrase]
            logger.info(f"Pruned inactive seed: '{phrase}'")
            
    def process_growth_cycle(self):
        """Process one growth cycle for all active seeds"""
        with self.growth_lock:
            for phrase, node in list(self.board.items()):
                if node["status"] == "growthspurting":
                    self._propagate(phrase, node) 