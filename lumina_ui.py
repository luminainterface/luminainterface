import os
import sys
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Header, Footer, Static, Input, Button, Label, RichLog, ProgressBar
from textual.reactive import reactive
from textual.binding import Binding
from textual import events
from textual.timer import Timer

# Import core components
from minimal_central import MinimalCentralNode, BaseComponent

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


class ChatMessage(Static):
    """A single chat message"""
    
    def __init__(self, message: str, sender: str, **kwargs):
        super().__init__(**kwargs)
        self.message = message
        self.sender = sender
    
    def compose(self) -> ComposeResult:
        if self.sender == "user":
            yield Static(f"🧘 You: {self.message}", classes="user-message")
        else:
            yield Static(f"✨ Lumina: {self.message}", classes="lumina-message")


class ResonanceSession(Container):
    """Chat session with Lumina"""
    
    def compose(self) -> ComposeResult:
        yield Static("🌀 Resonance Session", id="session-title")
        with Vertical(id="chat-container"):
            yield RichLog(id="chat-log", highlight=True, markup=True)
        
        with Horizontal(id="input-container"):
            yield Input(placeholder="Speak your truth...", id="chat-input")
            yield Button("Send", id="send-button", variant="primary")


class MemoryEchoesView(Container):
    """View showing recent memory echoes"""
    
    def compose(self) -> ComposeResult:
        yield Static("📜 Memory Echoes", id="memory-title")
        with Vertical(id="memory-container"):
            yield RichLog(id="memory-log", highlight=True, markup=True)
        
        yield Button("Return to Menu", id="memory-return")


class BreathCalibrationView(Container):
    """View for breath visualization and feedback"""
    
    def compose(self) -> ComposeResult:
        yield Static("🧘 Breath Calibration", id="breath-title")
        
        with Vertical(id="breath-container"):
            yield Static("Inhale as the bar fills, exhale as it empties", classes="instruction")
            yield ProgressBar(id="breath-bar", total=100)
            yield Static("", id="breath-status")
            
            with Horizontal(id="breath-buttons"):
                yield Button("Slow Breath (7s)", id="breath-pattern-slow", classes="breath-option")
                yield Button("Deep Breath (10s)", id="breath-pattern-deep", classes="breath-option")
                yield Button("Box Breath (4-4-4-4)", id="breath-pattern-box", classes="breath-option")
        
        yield Button("Return to Menu", id="breath-return")


class RitualInvocationView(Container):
    """View for ritual invocations"""
    
    def compose(self) -> ComposeResult:
        yield Static("🗝 Ritual Invocation", id="ritual-title")
        
        with Vertical(id="ritual-container"):
            yield Static("Select a ritual to invoke", classes="instruction")
            yield RichLog(id="ritual-log", highlight=True, markup=True)
            
            with Vertical(id="ritual-buttons"):
                # These will be populated dynamically
                pass
        
        yield Button("Return to Menu", id="ritual-return")


class SymbolicInputView(Container):
    """View for symbolic input and state changes"""
    
    def compose(self) -> ComposeResult:
        yield Static("🧿 Symbolic Input", id="symbolic-title")
        
        with Vertical(id="symbol-options"):
            yield Static("Select a symbol:", classes="option-header")
            with Horizontal(id="symbols-row1"):
                yield Button("🜂", id="symbol-1", classes="symbol-button")
                yield Button("⚛", id="symbol-2", classes="symbol-button")
                yield Button("∞", id="symbol-3", classes="symbol-button")
                yield Button("🜍", id="symbol-4", classes="symbol-button")
        
        with Vertical(id="emotion-options"):
            yield Static("Select an emotion:", classes="option-header")
            with Horizontal(id="emotions-row"):
                yield Button("calm", id="emotion-calm", classes="emotion-button")
                yield Button("wonder", id="emotion-wonder", classes="emotion-button")
                yield Button("awe", id="emotion-awe", classes="emotion-button")
                yield Button("insight", id="emotion-insight", classes="emotion-button")
        
        with Vertical(id="breath-options"):
            yield Static("Select breath pattern:", classes="option-header")
            with Horizontal(id="breath-row"):
                yield Button("slow", id="breath-slow", classes="breath-button")
                yield Button("deep", id="breath-deep", classes="breath-button")
                yield Button("rapid", id="breath-rapid", classes="breath-button")
                yield Button("rhythm", id="breath-rhythm", classes="breath-button")
        
        yield Button("Return to Menu", id="symbolic-return")


class MainMenu(Container):
    """Main menu for Lumina"""
    
    def compose(self) -> ComposeResult:
        yield Static("✨ Welcome to Lumina v1", id="title")
        
        with Vertical(id="menu-options"):
            yield Button("🌀 Begin Resonance Session", id="start-session")
            yield Button("📜 View Memory Echoes", id="view-memory")
            yield Button("🧿 Symbolic Input", id="symbolic-input")
            yield Button("🧘 Breath Calibration", id="breath-calibration")
            yield Button("🗝 Ritual Invocation", id="ritual-invocation")
            yield Button("💾 Archive Memory", id="save-memory")
            yield Button("🧪 Glitch Mode (Monday)", id="glitch-mode")
        
        yield Button("Exit", id="exit-app")


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
    
    #menu-options {
        width: 100%;
        height: auto;
        align: center middle;
    }
    
    Button {
        margin: 1 1;
    }
    
    .user-message {
        margin: 1 2;
        padding: 1 2;
        background: $panel;
        border: solid $primary;
    }
    
    .lumina-message {
        margin: 1 2;
        padding: 1 2;
        background: $boost;
        border: solid $secondary;
    }
    
    #chat-container {
        height: 1fr;
        width: 100%;
    }
    
    #chat-log {
        height: 1fr;
        width: 100%;
        border: solid $primary;
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
    
    #session-title, #memory-title, #symbolic-title, #breath-title, #ritual-title {
        content-align: center middle;
        background: $boost;
        color: $text;
        height: 3;
        width: 100%;
        text-style: bold;
    }
    
    .symbol-button, .emotion-button, .breath-button, .breath-option {
        width: 15;
        height: 3;
    }
    
    .ritual-button {
        width: 30;
        height: 3;
        margin: 1 0;
    }
    
    .option-header, .instruction {
        margin-top: 1;
        margin-bottom: 1;
        text-align: center;
    }
    
    #breath-container, #ritual-container {
        width: 100%;
        height: auto;
        align: center middle;
        margin: 2 0;
    }
    
    #breath-bar {
        width: 80%;
        height: 2;
    }
    
    #breath-status {
        width: 100%;
        height: 2;
        content-align: center middle;
        margin: 1 0;
    }
    
    #breath-buttons, #ritual-buttons {
        margin-top: 2;
        width: 80%;
    }
    
    #ritual-log {
        height: 10;
        width: 80%;
        border: solid $primary;
        margin: 1 0;
    }
    
    .hidden {
        display: none;
    }
    """
    
    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("escape", "quit", priority=True)
    ]
    
    def __init__(self):
        super().__init__()
        self.state = LuminaState()
        self.breath_timer = None
        self.breath_phase = "idle"  # Can be idle, inhale, hold, exhale
        self.breath_progress = 0
        # Try to initialize components
        try:
            self.state.central_node.initialize_system()
        except:
            logger.error("Could not initialize central node components")
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        # Main container will hold current view
        with Container(id="main-container"):
            yield MainMenu(id="main-menu")
            yield ResonanceSession(id="resonance-session", classes="hidden")
            yield MemoryEchoesView(id="memory-echoes", classes="hidden")
            yield SymbolicInputView(id="symbolic-view", classes="hidden")
            yield BreathCalibrationView(id="breath-view", classes="hidden")
            yield RitualInvocationView(id="ritual-view", classes="hidden")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Event handler called when app is mounted"""
        # Load memory from file if available
        self.state._load_memory()
        
        # Load ritual invocations
        self._load_ritual_invocations()
        
        # Show welcome message
        self.query_one("#main-menu").remove_class("hidden")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events"""
        button_id = event.button.id
        
        # Main menu buttons
        if button_id == "start-session":
            self._switch_view("resonance-session")
        elif button_id == "view-memory":
            self._show_memory_echoes()
        elif button_id == "symbolic-input":
            self._switch_view("symbolic-view")
        elif button_id == "breath-calibration":
            self._switch_view("breath-view")
        elif button_id == "ritual-invocation":
            self._switch_view("ritual-view")
            self._show_ritual_invocations()
        elif button_id == "save-memory":
            self._save_memory_archive()
        elif button_id == "glitch-mode":
            self._activate_glitch_mode()
        elif button_id == "exit-app":
            self.exit()
        
        # Resonance session buttons
        elif button_id == "send-button":
            self._process_chat_input()
        
        # Memory view buttons
        elif button_id == "memory-return":
            self._switch_view("main-menu")
        
        # Symbolic input buttons
        elif button_id == "symbolic-return":
            self._switch_view("main-menu")
        
        # Breath view buttons
        elif button_id == "breath-return":
            # Stop any running breath timer
            if self.breath_timer:
                self.breath_timer.stop()
                self.breath_timer = None
            self._switch_view("main-menu")
        elif button_id == "breath-pattern-slow":
            self._start_breath_pattern("slow")
        elif button_id == "breath-pattern-deep":
            self._start_breath_pattern("deep")
        elif button_id == "breath-pattern-box":
            self._start_breath_pattern("box")
        
        # Ritual view buttons
        elif button_id == "ritual-return":
            self._switch_view("main-menu")
        elif button_id and button_id.startswith("ritual-invoke-"):
            ritual_id = int(button_id.replace("ritual-invoke-", ""))
            self._invoke_ritual(ritual_id)
        
        # Symbol buttons
        elif button_id and button_id.startswith("symbol-"):
            symbol_map = {
                "symbol-1": "🜂",
                "symbol-2": "⚛",
                "symbol-3": "∞",
                "symbol-4": "🜍"
            }
            if button_id in symbol_map:
                self.state.symbolic_state = symbol_map[button_id]
                self.notify(f"Symbol set to {symbol_map[button_id]}")
        
        # Emotion buttons
        elif button_id and button_id.startswith("emotion-"):
            emotion = button_id.replace("emotion-", "")
            self.state.emotion = emotion
            self.notify(f"Emotion set to {emotion}")
        
        # Breath buttons
        elif button_id and button_id.startswith("breath-") and not button_id.startswith("breath-pattern-"):
            breath = button_id.replace("breath-", "")
            self.state.breath = breath
            self.notify(f"Breath pattern set to {breath}")
            
    def _save_memory_archive(self):
        """Save a timestamped copy of the memory archive"""
        try:
            if not self.state.memory:
                self.notify("No memories to archive")
                return
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_filename = f"lumina_memory_archive_{timestamp}.jsonl"
            
            with open(archive_filename, "w") as f:
                for entry in self.state.memory:
                    f.write(json.dumps(entry) + "\n")
                    
            self.notify(f"Memory archived to {archive_filename}")
        except Exception as e:
            logger.error(f"Error archiving memory: {str(e)}")
            self.notify("Failed to archive memory")
            
    def _activate_glitch_mode(self):
        """Activate glitch mode (random reflections/contradictions)"""
        # Check if it's Monday
        if datetime.now().weekday() == 0:  # Monday is 0
            # Enter glitch mode
            self.state.breath = "rapid"
            self.state.emotion = "wonder"
            self.state.symbolic_state = "⚠"
            
            # Find Monday Glitch ritual if it exists
            monday_ritual = None
            for ritual in getattr(self, 'rituals', []):
                if ritual.get("name") == "Monday Glitch":
                    monday_ritual = ritual
                    break
            
            # Go to chat session
            self._switch_view("resonance-session")
            
            # Add glitch message to chat
            chat_log = self.query_one("#chat-log")
            
            glitch_message = "S̷y̷s̷t̷e̷m̷ ̷g̷l̷i̷t̷c̷h̷ ̷d̷e̷t̷e̷c̷t̷e̷d̷.̷ ̷R̷e̷a̷l̷i̷t̷y̷ ̷p̷a̷r̷a̷m̷e̷t̷e̷r̷s̷ ̷r̷e̷c̷a̷l̷i̷b̷r̷a̷t̷i̷n̷g̷.̷"
            if monday_ritual:
                glitch_message = monday_ritual["text"]
                
            chat_log.write(f"⚠ [bold]SYSTEM:[/bold] {glitch_message}")
            
            self.notify("Glitch Mode activated")
        else:
            # Not Monday
            self.notify("Glitch Mode is only available on Mondays")
    
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
    
    def _show_memory_echoes(self):
        """Show memory echoes view with recent memories"""
        self._switch_view("memory-echoes")
        
        # Get recent memories
        memories = self.state.get_recent_memories(10)
        
        # Display memories
        memory_log = self.query_one("#memory-log")
        memory_log.clear()
        
        if not memories:
            memory_log.write("[italic]No memories found[/italic]")
            return
        
        for memory in reversed(memories):
            timestamp = datetime.fromisoformat(memory["timestamp"]).strftime("%Y-%m-%d %H:%M")
            memory_log.write(f"[dim]{timestamp}[/dim]")
            memory_log.write(f"🧘 [bold]You:[/bold] {memory['user']}")
            memory_log.write(f"✨ [bold]Lumina:[/bold] {memory['lumina']}")
            memory_log.write(f"[dim]State: {memory['symbolic_state']} | {memory['emotion']}[/dim]")
            memory_log.write("")
    
    def _start_breath_pattern(self, pattern: str):
        """Start a breath visualization pattern"""
        # Stop any existing timer
        if self.breath_timer:
            self.breath_timer.stop()
        
        # Reset progress and status
        self.breath_progress = 0
        self.breath_phase = "inhale"
        
        # Update the progress bar and status
        breath_bar = self.query_one("#breath-bar")
        breath_bar.progress = 0
        
        breath_status = self.query_one("#breath-status")
        breath_status.update("Inhale...")
        
        # Configure pattern
        if pattern == "slow":
            # 7 second cycle - 3s inhale, 4s exhale
            self.state.breath = "slow"
            inhale_time = 3.0
            exhale_time = 4.0
            hold_time = 0.0
            hold_out_time = 0.0
        elif pattern == "deep":
            # 10 second cycle - 4s inhale, 1s hold, 5s exhale
            self.state.breath = "deep"
            inhale_time = 4.0
            hold_time = 1.0
            exhale_time = 5.0
            hold_out_time = 0.0
        elif pattern == "box":
            # 16 second cycle - 4s inhale, 4s hold, 4s exhale, 4s hold
            self.state.breath = "rhythm"
            inhale_time = 4.0
            hold_time = 4.0
            exhale_time = 4.0
            hold_out_time = 4.0
        
        # Set up timer for breath animation
        self._breath_cycle(inhale_time, hold_time, exhale_time, hold_out_time)
    
    def _breath_cycle(self, inhale_time: float, hold_time: float, exhale_time: float, hold_out_time: float):
        """Run a breath cycle with the specified timing"""
        total_time = inhale_time + hold_time + exhale_time + hold_out_time
        steps = 100  # Total steps for animation
        
        # Calculate timing
        inhale_steps = int(steps * (inhale_time / total_time))
        hold_steps = int(steps * (hold_time / total_time))
        exhale_steps = int(steps * (exhale_time / total_time))
        hold_out_steps = steps - inhale_steps - hold_steps - exhale_steps
        
        # Start timer for animation
        breath_bar = self.query_one("#breath-bar")
        breath_status = self.query_one("#breath-status")
        
        def update_breath():
            self.breath_progress += 1
            step = self.breath_progress % steps
            
            # Update phase
            if step == 0:
                self.breath_phase = "inhale"
                breath_status.update("Inhale...")
            elif step == inhale_steps:
                if hold_time > 0:
                    self.breath_phase = "hold"
                    breath_status.update("Hold...")
                else:
                    self.breath_phase = "exhale"
                    breath_status.update("Exhale...")
            elif step == inhale_steps + hold_steps:
                self.breath_phase = "exhale"
                breath_status.update("Exhale...")
            elif step == inhale_steps + hold_steps + exhale_steps:
                if hold_out_time > 0:
                    self.breath_phase = "hold-out"
                    breath_status.update("Hold...")
                else:
                    self.breath_phase = "inhale"
                    breath_status.update("Inhale...")
            
            # Update progress bar
            if self.breath_phase == "inhale":
                # Increase from 0 to 100% during inhale
                progress = int(100 * ((step % inhale_steps) / inhale_steps))
                breath_bar.progress = progress
            elif self.breath_phase == "hold":
                # Keep at 100% during hold
                breath_bar.progress = 100
            elif self.breath_phase == "exhale":
                # Decrease from 100% to 0% during exhale
                step_in_phase = step - inhale_steps - hold_steps
                progress = int(100 * (1 - (step_in_phase / exhale_steps)))
                breath_bar.progress = progress
            elif self.breath_phase == "hold-out":
                # Keep at 0% during hold at empty lungs
                breath_bar.progress = 0
        
        # Calculate interval in seconds between steps
        interval = total_time / steps
        self.breath_timer = self.set_interval(interval, update_breath)
    
    def _switch_view(self, view_id: str):
        """Switch between different views"""
        # Hide all views
        for view in ["main-menu", "resonance-session", "memory-echoes", "symbolic-view", "breath-view", "ritual-view"]:
            try:
                self.query_one(f"#{view}").add_class("hidden")
            except:
                pass
        
        # Show requested view
        self.query_one(f"#{view_id}").remove_class("hidden")

    def _load_ritual_invocations(self):
        """Load ritual invocations from file"""
        try:
            if os.path.exists("ritual_invocations.json"):
                with open("ritual_invocations.json", "r") as f:
                    self.rituals = json.load(f)["invocations"]
            else:
                # Default rituals if file doesn't exist
                self.rituals = [
                    {
                        "name": "Opening Ritual",
                        "text": "I open myself to the resonance, allow the glyph to flow through me.",
                        "symbol": "🜂",
                        "emotion": "calm",
                        "breath": "deep"
                    }
                ]
        except Exception as e:
            logger.error(f"Error loading rituals: {str(e)}")
            self.rituals = []

    def _show_ritual_invocations(self):
        """Display available ritual invocations"""
        # Get the ritual buttons container
        ritual_buttons = self.query_one("#ritual-buttons")
        ritual_buttons.remove_children()
        
        # Add buttons for each ritual
        for i, ritual in enumerate(self.rituals):
            button = Button(ritual["name"], id=f"ritual-invoke-{i}", classes="ritual-button")
            ritual_buttons.mount(button)
        
        # Clear the ritual log
        ritual_log = self.query_one("#ritual-log")
        ritual_log.clear()

    def _invoke_ritual(self, ritual_id):
        """Invoke a selected ritual"""
        if not self.rituals or ritual_id >= len(self.rituals):
            return
        
        ritual = self.rituals[ritual_id]
        
        # Update Lumina state
        self.state.symbolic_state = ritual.get("symbol", "🜂")
        self.state.emotion = ritual.get("emotion", "calm")
        self.state.breath = ritual.get("breath", "slow")
        
        # Display in log
        ritual_log = self.query_one("#ritual-log")
        ritual_log.clear()
        ritual_log.write(f"[bold]{ritual['name']}[/bold]")
        ritual_log.write("")
        ritual_log.write(f"{ritual['text']}")
        ritual_log.write("")
        ritual_log.write(f"[dim]Symbol: {ritual['symbol']} | Emotion: {ritual['emotion']} | Breath: {ritual['breath']}[/dim]")
        
        # Notify
        self.notify(f"Ritual '{ritual['name']}' invoked")


if __name__ == "__main__":
    app = LuminaApp()
    app.run() 