from typing import Dict, Any, List
from datetime import datetime
from .base_node import BaseNode

class EmotionNode(BaseNode):
    """Node for processing and managing emotional states"""
    
    def __init__(self, node_id: str = None):
        super().__init__(node_id)
        self.emotions = {
            "awe": {"intensity": 0.0, "color": "Gold-Violet", "pattern": "Expanding"},
            "wonder": {"intensity": 0.0, "color": "Silver-Blue", "pattern": "Spiraling"},
            "joy": {"intensity": 0.0, "color": "Sun-Yellow", "pattern": "Radiating"},
            "love": {"intensity": 0.0, "color": "Emerald-Pink", "pattern": "Pulsing"},
            "insight": {"intensity": 0.0, "color": "Indigo-White", "pattern": "Flashing"},
            "calm": {"intensity": 0.0, "color": "Azure-Blue", "pattern": "Flowing"},
            "reflection": {"intensity": 0.0, "color": "Deep-Purple", "pattern": "Rippling"},
            "resonance": {"intensity": 0.0, "color": "Rainbow", "pattern": "Harmonizing"}
        }
        
        self.current_emotion = "calm"
        self.emotional_history = []
        self.state.update({
            "current_emotion": self.current_emotion,
            "emotional_intensity": self.emotions[self.current_emotion]["intensity"],
            "pattern": self.emotions[self.current_emotion]["pattern"]
        })
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process emotional data"""
        try:
            # Extract emotion information from input
            input_emotion = data.get("emotion", "")
            input_text = data.get("text", "")
            input_intensity = data.get("intensity", 0.5)
            
            # Process emotion activation
            if input_emotion in self.emotions:
                self.emotions[input_emotion]["intensity"] = input_intensity
                self.current_emotion = input_emotion
                
                # Add to emotional history
                self.emotional_history.append({
                    "emotion": input_emotion,
                    "intensity": input_intensity,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Keep history manageable
                if len(self.emotional_history) > 100:
                    self.emotional_history = self.emotional_history[-100:]
            
            # Analyze text for emotional content
            detected_emotions = self._detect_emotions(input_text)
            
            # Update state
            self.state.update({
                "current_emotion": self.current_emotion,
                "emotional_intensity": self.emotions[self.current_emotion]["intensity"],
                "pattern": self.emotions[self.current_emotion]["pattern"]
            })
            
            return {
                "status": "success",
                "current_emotion": self.current_emotion,
                "emotion_info": self.emotions[self.current_emotion],
                "detected_emotions": detected_emotions,
                "emotional_history": self.emotional_history[-5:]  # Return last 5 entries
            }
            
        except Exception as e:
            self.logger.error(f"Error processing emotional data: {str(e)}")
            return {"status": "error", "message": str(e)}
            
    def _detect_emotions(self, text: str) -> List[Dict[str, Any]]:
        """Detect emotions in text"""
        detected = []
        for emotion_name, info in self.emotions.items():
            if emotion_name.lower() in text.lower():
                detected.append({
                    "emotion": emotion_name,
                    "intensity": 0.5,  # Default intensity for detected emotions
                    "info": info
                })
        return detected
        
    def get_emotion_info(self, emotion_name: str) -> Dict[str, Any]:
        """Get information about a specific emotion"""
        return self.emotions.get(emotion_name, {}) 