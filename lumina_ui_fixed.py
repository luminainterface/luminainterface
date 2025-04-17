"""
Lumina UI - A text-based interface for the Lumina Neural Network System
Modified to be compatible with Textual 3.1.0
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Header, Footer, Static, Input, Button, RichLog, ProgressBar
from textual.binding import Binding

# Import core components
try:
    from minimal_central import MinimalCentralNode, BaseComponent
except ImportError:
    # Create mock classes if imports fail
    class MinimalCentralNode:
        def __init__(self):
            self.component_registry = {}
        
        def process_complete_flow(self, data):
            return {
                "action": "respond",
                "glyph": "✨",
                "story": "Resonating with your input.",
                "signal": 0.85
            }
    
    class BaseComponent:
        def __init__(self):
            pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("lumina.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LuminaSystem")

class LuminaState:
    """Manages the state of the Lumina system"""
    
    def __init__(self):
        self.emotion = "calm"
        self.symbolic_state = "🜂"
        self.breath = "slow"
        self.memory = []
        self.mirror = False
        self.central_node = MinimalCentralNode()
        
    def add_memory(self, user_input: str, lumina_response: str):
        """Add a memory entry"""
        entry = {
            "user": user_input,
            "lumina": lumina_response,
            "timestamp": datetime.now().isoformat(),
            "emotion": self.emotion,
            "symbolic_state": self.symbolic_state
        }
        self.memory.append(entry)
        self._save_memory()
        
    def get_recent_memories(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get recent memory entries"""
        return self.memory[-count:] if self.memory else []
    
    def process_input(self, user_input: str) -> str:
        """Process user input and generate response"""
        # Extract symbols, emotions and other parameters if they are included
        input_data = self._parse_input(user_input)
        
        # Update state based on input 
        self._update_state(input_data)
        
        # Process through central node if available
        try:
            result = self.central_node.process_complete_flow(input_data)
            response = self._format_response(result)
        except Exception as e:
            logger.error(f"Error processing input: {str(e)}")
            response = "I sense a disturbance in the field. Let's recalibrate."
        
        # Add to memory
        self.add_memory(user_input, response)
        
        return response
    
    def _parse_input(self, text: str) -> Dict[str, Any]:
        """Parse user input into structured data"""
        input_data = {
            "text": text,
            "symbol": self.symbolic_state,
            "emotion": self.emotion,
            "breath": self.breath,
            "paradox": None
        }
        
        # Check for explicit symbol notation like :infinity:
        if ":" in text:
            parts = text.split(":")
            for i in range(1, len(parts), 2):
                if i < len(parts) - 1:
                    key = parts[i].strip().lower()
                    value = parts[i+1].strip()
                    if key in ["symbol", "emotion", "breath", "paradox"]:
                        input_data[key] = value
                        # Remove from main text
                        text = text.replace(f":{key}:{value}", "").strip()
            
            # Update the cleaned text
            input_data["text"] = text
            
        return input_data
    
    def _update_state(self, input_data: Dict[str, Any]):
        """Update internal state based on input"""
        # Update symbolic state if provided
        if input_data.get("symbol"):
            self.symbolic_state = input_data["symbol"]
            
        # Update emotion if provided
        if input_data.get("emotion"):
            self.emotion = input_data["emotion"]
            
        # Update breath if provided
        if input_data.get("breath"):
            self.breath = input_data["breath"]
    
    def _format_response(self, result: Dict[str, Any]) -> str:
        """Format central node result into a response string"""
        # If there's a story, use that as the primary response
        if result.get("story"):
            return result["story"]
            
        # Otherwise create a composite response
        response_parts = []
        
        if result.get("action"):
            response_parts.append(f"{result['action']}")
            
        if result.get("glyph"):
            response_parts.append(f"{result['glyph']}")
            
        if result.get("signal"):
            response_parts.append(f"Signal strength: {result['signal']}")
            
        if not response_parts:
            return "I'm listening to the resonance. Tell me more."
            
        return " ".join(response_parts)
    
    def _save_memory(self):
        """Save memory to a file"""
        try:
            with open("lumina_memory.jsonl", "w") as f:
                for entry in self.memory:
                    f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Error saving memory: {str(e)}")
    
    def _load_memory(self):
        """Load memory from file"""
        try:
            if os.path.exists("lumina_memory.jsonl"):
                with open("lumina_memory.jsonl", "r") as f:
                    self.memory = [json.loads(line) for line in f if line.strip()]
        except Exception as e:
            logger.error(f"Error loading memory: {str(e)}")


class MainScreen(Container):
    """Main screen with chat interface"""
    
    def compose(self) -> ComposeResult:
        yield Static("✨ Lumina v1", id="title")
        
        with Vertical(id="chat-container"):
            yield RichLog(id="chat-log", markup=True)
            
        with Horizontal(id="input-container"):
            yield Input(placeholder="Speak your truth...", id="chat-input")
            yield Button("Send", id="send-button", variant="primary")


class LuminaApp(App):
    """Main Lumina application with text-based UI"""
    
    CSS = """
    #title {
        dock: top;
        content-align: center middle;
        background: $boost;
        color: $text;
        height: 3;
        width: 100%;
        text-style: bold;
    }
    
    #chat-container {
        height: 1fr;
        width: 100%;
    }
    
    #chat-log {
        height: 1fr;
        width: 100%;
        border: solid $primary;
        padding: 1 2;
    }
    
    #input-container {
        height: 3;
        width: 100%;
        align: center middle;
    }
    
    #chat-input {
        width: 4fr;
    }
    
    #send-button {
        width: 1fr;
    }
    """
    
    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("escape", "quit", priority=True)
    ]
    
    def __init__(self):
        super().__init__()
        self.state = LuminaState()
    
    def compose(self) -> ComposeResult:
        yield Header()
        yield MainScreen()
        yield Footer()
    
    def on_mount(self) -> None:
        """Event handler called when app is mounted"""
        # Load memory from file if available
        self.state._load_memory()
        
        # Show welcome message
        self.query_one("#chat-log").write("✨ [bold]Lumina:[/bold] Welcome. I am listening to the resonance. Tell me more.")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events"""
        if event.button.id == "send-button":
            self._process_chat_input()
    
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission events"""
        if event.input.id == "chat-input":
            self._process_chat_input()
    
    def _process_chat_input(self):
        """Process input from chat box"""
        chat_input = self.query_one("#chat-input")
        user_input = chat_input.value.strip()
        
        if not user_input:
            return
        
        # Add user message to chat
        chat_log = self.query_one("#chat-log")
        chat_log.write(f"🧘 [bold]You:[/bold] {user_input}")
        
        # Process input
        response = self.state.process_input(user_input)
        
        # Add Lumina's response to chat
        chat_log.write(f"✨ [bold]Lumina:[/bold] {response}")
        
        # Clear input
        chat_input.value = ""


if __name__ == "__main__":
    app = LuminaApp()
    app.run() 