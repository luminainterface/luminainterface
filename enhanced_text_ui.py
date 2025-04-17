"""
Enhanced Lumina UI - An improved text-based interface for the Lumina Neural Network System
Compatible with Textual 3.1.0
"""

import os
import sys
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
import uuid

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, Horizontal, Grid
from textual.widgets import (
    Header, Footer, Static, Input, Button, RichLog, 
    ProgressBar, Label, TextLog, Tree
)
from textual.binding import Binding
from textual import events
from textual.widgets._tree import TreeNode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("lumina_ui.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LuminaUI")

# Default keybindings configuration
DEFAULT_KEYBINDINGS = {
    "quit": ["q", "escape"],
    "send_message": ["enter"],
    "command_history_up": ["up"],
    "command_history_down": ["down"],
    "clear_input": ["ctrl+k"],
    "clear_log": ["ctrl+l"],
    "focus_input": ["ctrl+i"],
    "toggle_memory_view": ["ctrl+m"],
    "toggle_glyph_panel": ["ctrl+g"],
    "help": ["f1"]
}

# Standard glyphs with Unicode representations
GLYPHS = {
    "fire": "🜂",
    "air": "🜁",
    "earth": "🜃",
    "water": "🜄",
    "salt": "🜔",
    "paradox": "🝊",
    "void": "🜚",
    "infinity": "∞",
    "star": "✦",
    "cross": "✠",
    "spiral": "⌀",
    "resonance": "≈",
    "mirror": "◊"
}

class Plugin:
    """Base class for UI plugins"""
    
    def __init__(self, app):
        self.app = app
        self.name = self.__class__.__name__
        self.enabled = True
    
    def on_load(self) -> None:
        """Called when the plugin is loaded"""
        logger.info(f"Loaded plugin: {self.name}")
    
    def on_unload(self) -> None:
        """Called when the plugin is unloaded"""
        logger.info(f"Unloaded plugin: {self.name}")
    
    def on_message_sent(self, message: str, response: Dict[str, Any]) -> None:
        """Called when a message is sent to Lumina"""
        pass
    
    def on_memory_updated(self) -> None:
        """Called when memory is updated"""
        pass


class PluginManager:
    """Manages plugin loading and callbacks"""
    
    def __init__(self, app):
        self.app = app
        self.plugins: Dict[str, Plugin] = {}
        self.plugin_dir = Path("plugins")
        self.plugin_dir.mkdir(exist_ok=True)
    
    def load_plugins(self) -> None:
        """Load all plugins from the plugins directory"""
        # For initial implementation, we'll just work with predefined plugins
        logger.info("Loading plugins...")
    
    def register_plugin(self, plugin: Plugin) -> None:
        """Register a plugin"""
        self.plugins[plugin.name] = plugin
        plugin.on_load()
        logger.info(f"Registered plugin: {plugin.name}")
    
    def trigger_event(self, event_name: str, *args, **kwargs) -> None:
        """Trigger an event on all enabled plugins"""
        method_name = f"on_{event_name}"
        for plugin_name, plugin in self.plugins.items():
            if plugin.enabled and hasattr(plugin, method_name):
                try:
                    getattr(plugin, method_name)(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in plugin {plugin_name}.{method_name}: {str(e)}")


class LuminaState:
    """Manages the state of the Lumina system"""
    
    def __init__(self):
        self.memory_file = "lumina_memory.jsonl"
        self.memory = []
        self.command_history = []
        self.command_index = -1
        self.session_id = str(uuid.uuid4())
        self.current_state = {
            "symbol": None,
            "emotion": None,
            "breath": None,
            "paradox": None
        }
        self._load_memory()
    
    def _load_memory(self) -> None:
        """Load memory from file"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.strip():
                            try:
                                memory_item = json.loads(line)
                                self.memory.append(memory_item)
                            except json.JSONDecodeError:
                                logger.error(f"Error parsing memory entry: {line}")
                logger.info(f"Loaded {len(self.memory)} memory entries")
            except Exception as e:
                logger.error(f"Error loading memory: {str(e)}")
        else:
            logger.info("No memory file found, starting with empty memory")
    
    def _save_memory(self) -> None:
        """Save memory to file"""
        try:
            with open(self.memory_file, "w") as f:
                for item in self.memory:
                    f.write(json.dumps(item) + "\n")
            logger.info(f"Saved {len(self.memory)} memory entries")
        except Exception as e:
            logger.error(f"Error saving memory: {str(e)}")
    
    def add_memory(self, user_input: str, system_response: Dict[str, Any], metadata: Dict[str, Any] = None) -> None:
        """Add a memory entry"""
        # Create memory entry
        memory_item = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "user_input": user_input,
            "system_response": system_response,
            "state": self.current_state.copy(),
            "metadata": metadata or {}
        }
        
        # Add to memory
        self.memory.append(memory_item)
        
        # Save memory
        self._save_memory()
    
    def add_to_command_history(self, command: str) -> None:
        """Add a command to the history"""
        # Don't add empty commands or duplicates of the last command
        if not command or (self.command_history and self.command_history[-1] == command):
            return
            
        self.command_history.append(command)
        if len(self.command_history) > 100:  # Limit history size
            self.command_history.pop(0)
        self.command_index = len(self.command_history)
    
    def get_previous_command(self) -> Optional[str]:
        """Get the previous command from history"""
        if not self.command_history or self.command_index <= 0:
            return None
            
        self.command_index = max(0, self.command_index - 1)
        return self.command_history[self.command_index]
    
    def get_next_command(self) -> Optional[str]:
        """Get the next command from history"""
        if not self.command_history or self.command_index >= len(self.command_history) - 1:
            self.command_index = len(self.command_history)
            return ""
            
        self.command_index += 1
        return self.command_history[self.command_index]
    
    def update_state(self, **kwargs) -> None:
        """Update the current state"""
        for key, value in kwargs.items():
            if key in self.current_state:
                self.current_state[key] = value
    
    def get_state_description(self) -> str:
        """Get a description of the current state"""
        active_states = []
        
        for key, value in self.current_state.items():
            if value:
                # Get glyph for key if available
                glyph = GLYPHS.get(value, "") if key == "symbol" else ""
                active_states.append(f"{key}: {value} {glyph}")
        
        if active_states:
            return " | ".join(active_states)
        else:
            return "No active state parameters"


class MemoryVisualizer(Container):
    """Visualizes memory entries in a tree view"""
    
    def __init__(self, state: LuminaState):
        super().__init__(id="memory-visualizer")
        self.state = state
    
    def compose(self) -> ComposeResult:
        yield Static("Memory Spiral", id="memory-title")
        yield Tree("Sessions", id="memory-tree")
    
    def on_mount(self) -> None:
        """Set up the memory tree when mounted"""
        self.update_memory_tree()
    
    def update_memory_tree(self) -> None:
        """Update the memory tree with current memory entries"""
        tree = self.query_one(Tree)
        tree.clear()
        
        # Group memory entries by session and date
        memory_by_session = {}
        for entry in self.state.memory:
            session_id = entry.get("session_id", "unknown")
            timestamp = entry.get("timestamp", "")
            date = timestamp.split("T")[0] if timestamp and "T" in timestamp else "unknown"
            
            if session_id not in memory_by_session:
                memory_by_session[session_id] = {}
                
            if date not in memory_by_session[session_id]:
                memory_by_session[session_id][date] = []
                
            memory_by_session[session_id][date].append(entry)
        
        # Add sessions to tree
        for session_id, dates in memory_by_session.items():
            session_label = f"Session {session_id[:8]}"
            if session_id == self.state.session_id:
                session_label += " (current)"
                
            session_node = tree.root.add(session_label, expanded=session_id == self.state.session_id)
            
            # Add dates to session
            for date, entries in dates.items():
                date_label = f"{date} ({len(entries)} entries)"
                date_node = session_node.add(date_label, expanded=session_id == self.state.session_id)
                
                # Add entries to date
                for i, entry in enumerate(entries):
                    user_input = entry.get("user_input", "")
                    timestamp = entry.get("timestamp", "").split("T")[1][:8] if entry.get("timestamp") else ""
                    
                    if len(user_input) > 30:
                        user_input = user_input[:27] + "..."
                        
                    entry_label = f"{timestamp} - {user_input}"
                    entry_node = date_node.add(entry_label)
                    entry_node.data = entry


class GlyphPanel(Container):
    """Panel for displaying and selecting glyphs"""
    
    def __init__(self, on_glyph_select=None):
        super().__init__(id="glyph-panel")
        self.on_glyph_select = on_glyph_select
    
    def compose(self) -> ComposeResult:
        yield Static("Symbolic Glyphs", id="glyph-title")
        
        with Grid(id="glyph-grid"):
            for glyph_name, glyph_symbol in GLYPHS.items():
                yield Button(f"{glyph_symbol} {glyph_name}", classes="glyph-button", id=f"glyph-{glyph_name}")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events"""
        button_id = event.button.id
        if button_id and button_id.startswith("glyph-"):
            glyph_name = button_id[6:]  # Remove "glyph-" prefix
            if self.on_glyph_select:
                self.on_glyph_select(glyph_name)


class EnhancedChatView(Container):
    """Enhanced chat interface with memory visualization and glyph panel"""
    
    def __init__(self, state: LuminaState, process_callback=None):
        super().__init__(id="chat-view")
        self.state = state
        self.process_callback = process_callback
        self.memory_visible = False
        self.glyphs_visible = False
    
    def compose(self) -> ComposeResult:
        yield Static("Lumina v1 - Enhanced", id="title")
        
        with Horizontal(id="main-container"):
            with Vertical(id="left-panel", classes="hidden"):
                yield MemoryVisualizer(self.state)
                
            with Vertical(id="center-panel"):
                yield RichLog(id="chat-log", markup=True)
                
                with Horizontal(id="state-container"):
                    yield Static("", id="state-display")
                
                with Horizontal(id="input-container"):
                    yield Input(placeholder="Speak your truth... (type /help for commands)", id="chat-input")
                    yield Button("Send", id="send-button", variant="primary")
            
            with Vertical(id="right-panel", classes="hidden"):
                yield GlyphPanel(on_glyph_select=self.on_glyph_select)
    
    def on_mount(self) -> None:
        """Set up the UI when mounted"""
        # Update state display
        self.update_state_display()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events"""
        if event.button.id == "send-button":
            self._process_chat_input()
    
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission events"""
        if event.input.id == "chat-input":
            self._process_chat_input()
    
    def _process_chat_input(self) -> None:
        """Process the chat input"""
        input_widget = self.query_one("#chat-input")
        user_input = input_widget.value.strip()
        
        if not user_input:
            return
        
        # Add to command history
        self.state.add_to_command_history(user_input)
        
        # Process commands
        if user_input.startswith("/"):
            self._process_command(user_input)
            input_widget.value = ""
            return
        
        # Display user message
        chat_log = self.query_one("#chat-log")
        chat_log.write(f"[bold blue]You:[/bold blue] {user_input}")
        
        # Process message
        if self.process_callback:
            response = self.process_callback(user_input)
            
            # Display response
            output_text = response.get("output", "")
            glyph = response.get("glyph", "")
            
            if glyph and glyph in GLYPHS.values():
                response_prefix = f"[bold purple]{glyph} Lumina:[/bold purple]"
            else:
                response_prefix = "[bold purple]Lumina:[/bold purple]"
                
            chat_log.write(f"{response_prefix} {output_text}")
            
            # Update state if parameters were provided
            if "parameters" in response:
                for param, value in response["parameters"].items():
                    if param in self.state.current_state:
                        self.state.update_state(**{param: value})
                self.update_state_display()
            
            # Add to memory
            self.state.add_memory(user_input, response)
            
            # Update memory visualizer if visible
            if self.memory_visible:
                self.query_one(MemoryVisualizer).update_memory_tree()
        
        # Clear input
        input_widget.value = ""
    
    def _process_command(self, command: str) -> None:
        """Process a command"""
        chat_log = self.query_one("#chat-log")
        parts = command.split()
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        if cmd == "/help":
            self._show_help()
        elif cmd == "/clear":
            chat_log.clear()
        elif cmd == "/memory":
            self.toggle_memory_view()
        elif cmd == "/glyphs":
            self.toggle_glyph_panel()
        elif cmd == "/state":
            self._show_state()
        elif cmd == "/symbol" and args:
            self.state.update_state(symbol=args[0])
            self.update_state_display()
            chat_log.write(f"[bold green]System:[/bold green] Set symbol to {args[0]}")
        elif cmd == "/emotion" and args:
            self.state.update_state(emotion=args[0])
            self.update_state_display()
            chat_log.write(f"[bold green]System:[/bold green] Set emotion to {args[0]}")
        elif cmd == "/breath" and args:
            self.state.update_state(breath=args[0])
            self.update_state_display()
            chat_log.write(f"[bold green]System:[/bold green] Set breath to {args[0]}")
        elif cmd == "/paradox" and args:
            self.state.update_state(paradox=args[0])
            self.update_state_display()
            chat_log.write(f"[bold green]System:[/bold green] Set paradox to {args[0]}")
        elif cmd == "/reset":
            self.state.update_state(symbol=None, emotion=None, breath=None, paradox=None)
            self.update_state_display()
            chat_log.write(f"[bold green]System:[/bold green] Reset all state parameters")
        else:
            chat_log.write(f"[bold red]System:[/bold red] Unknown command: {cmd}")
    
    def _show_help(self) -> None:
        """Show help information"""
        chat_log = self.query_one("#chat-log")
        chat_log.write("[bold green]Available Commands:[/bold green]")
        chat_log.write("  [bold]/help[/bold] - Show this help information")
        chat_log.write("  [bold]/clear[/bold] - Clear the chat log")
        chat_log.write("  [bold]/memory[/bold] - Toggle memory visualization")
        chat_log.write("  [bold]/glyphs[/bold] - Toggle glyph panel")
        chat_log.write("  [bold]/state[/bold] - Show current state parameters")
        chat_log.write("  [bold]/symbol <value>[/bold] - Set symbol parameter")
        chat_log.write("  [bold]/emotion <value>[/bold] - Set emotion parameter")
        chat_log.write("  [bold]/breath <value>[/bold] - Set breath parameter")
        chat_log.write("  [bold]/paradox <value>[/bold] - Set paradox parameter")
        chat_log.write("  [bold]/reset[/bold] - Reset all state parameters")
        chat_log.write("")
        chat_log.write("[bold green]Special Input Format:[/bold green]")
        chat_log.write("  You can include parameters in your message using the format:")
        chat_log.write("  [bold]:parameter:value[/bold]")
        chat_log.write("  Example: [bold]Tell me about consciousness :symbol:infinity[/bold]")
    
    def _show_state(self) -> None:
        """Show current state parameters"""
        chat_log = self.query_one("#chat-log")
        chat_log.write("[bold green]Current State Parameters:[/bold green]")
        
        for key, value in self.state.current_state.items():
            glyph = ""
            if key == "symbol" and value in GLYPHS:
                glyph = f" {GLYPHS[value]}"
                
            value_str = value if value is not None else "None"
            chat_log.write(f"  [bold]{key}:[/bold] {value_str}{glyph}")
    
    def toggle_memory_view(self) -> None:
        """Toggle memory visualization panel"""
        left_panel = self.query_one("#left-panel")
        if "hidden" in left_panel.classes:
            left_panel.remove_class("hidden")
            self.memory_visible = True
            # Update the memory tree
            self.query_one(MemoryVisualizer).update_memory_tree()
        else:
            left_panel.add_class("hidden")
            self.memory_visible = False
    
    def toggle_glyph_panel(self) -> None:
        """Toggle glyph panel"""
        right_panel = self.query_one("#right-panel")
        if "hidden" in right_panel.classes:
            right_panel.remove_class("hidden")
            self.glyphs_visible = True
        else:
            right_panel.add_class("hidden")
            self.glyphs_visible = False
    
    def on_key(self, event: events.Key) -> None:
        """Handle key events"""
        # Check for command history navigation
        if event.key == "up":
            prev_command = self.state.get_previous_command()
            if prev_command is not None:
                input_widget = self.query_one("#chat-input")
                input_widget.value = prev_command
                # Move cursor to end of input
                input_widget.action_end()
        elif event.key == "down":
            next_command = self.state.get_next_command()
            if next_command is not None:
                input_widget = self.query_one("#chat-input")
                input_widget.value = next_command
                # Move cursor to end of input
                input_widget.action_end()
    
    def update_state_display(self) -> None:
        """Update the state display"""
        state_display = self.query_one("#state-display")
        state_display.update(self.state.get_state_description())
    
    def on_glyph_select(self, glyph_name: str) -> None:
        """Handle glyph selection"""
        self.state.update_state(symbol=glyph_name)
        self.update_state_display()
        
        chat_log = self.query_one("#chat-log")
        glyph_symbol = GLYPHS.get(glyph_name, "")
        chat_log.write(f"[bold green]System:[/bold green] Set symbol to {glyph_name} {glyph_symbol}")


class EnhancedLuminaApp(App):
    """Enhanced Lumina application with text-based UI"""
    
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
    
    #main-container {
        height: 1fr;
        width: 100%;
    }
    
    #center-panel {
        height: 1fr;
        width: 3fr;
    }
    
    #left-panel {
        height: 1fr;
        width: 1fr;
        background: $surface;
        display: block;
    }
    
    #right-panel {
        height: 1fr;
        width: 1fr;
        background: $surface;
        display: block;
    }
    
    .hidden {
        display: none;
        width: 0;
    }
    
    #chat-log {
        height: 1fr;
        width: 100%;
        border: solid $primary;
        padding: 1 2;
        min-height: 20;
    }
    
    #state-container {
        height: 1;
        width: 100%;
    }
    
    #state-display {
        width: 100%;
        height: 1;
        background: $panel;
        color: $secondary;
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
    
    #memory-visualizer {
        width: 100%;
        height: 1fr;
    }
    
    #memory-title, #glyph-title {
        content-align: center middle;
        background: $boost;
        color: $text;
        height: 2;
        width: 100%;
        text-style: bold;
    }
    
    #memory-tree {
        height: 1fr;
        width: 100%;
        border: solid $primary;
        padding: 0 1;
    }
    
    #glyph-panel {
        width: 100%;
        height: 1fr;
    }
    
    #glyph-grid {
        width: 100%;
        height: auto;
        grid-size: 2;
        grid-gutter: 1 1;
        padding: 1;
    }
    
    .glyph-button {
        width: 100%;
        height: 3;
    }
    """
    
    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("escape", "quit", priority=True),
        Binding("ctrl+m", "toggle_memory", "Toggle Memory"),
        Binding("ctrl+g", "toggle_glyphs", "Toggle Glyphs"),
        Binding("ctrl+l", "clear_log", "Clear Log"),
        Binding("f1", "show_help", "Help")
    ]
    
    def __init__(self):
        super().__init__()
        self.state = LuminaState()
        self.plugin_manager = PluginManager(self)
    
    def compose(self) -> ComposeResult:
        yield Header()
        yield EnhancedChatView(self.state, self.process_message)
        yield Footer()
    
    def on_mount(self) -> None:
        """Event handler called when app is mounted"""
        # Load plugins
        self.plugin_manager.load_plugins()
        
        # Focus input
        self.query_one("#chat-input").focus()
    
    def process_message(self, message: str) -> Dict[str, Any]:
        """Process a message through the central node or minimal implementation"""
        try:
            # Here we would normally connect to the central node
            # For this example, we'll use a minimal implementation
            response = {
                "output": f"I processed your message: '{message}'",
                "glyph": "✨",
                "action": "respond",
                "story": "Resonating with your input.",
                "signal": 0.85
            }
            
            # Extract parameters from message
            params = {}
            words = message.split()
            clean_words = []
            
            for word in words:
                if word.startswith(":") and ":" in word[1:]:
                    param, value = word[1:].split(":", 1)
                    params[param] = value
                else:
                    clean_words.append(word)
            
            if params:
                response["parameters"] = params
                
                # Apply symbol glyph if provided
                if "symbol" in params and params["symbol"] in GLYPHS:
                    response["glyph"] = GLYPHS[params["symbol"]]
            
            # Trigger plugin event
            self.plugin_manager.trigger_event("message_sent", message, response)
            
            return response
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return {
                "output": "I'm sorry, I encountered an error processing your message.",
                "glyph": "⚠",
                "status": "error"
            }
    
    def action_toggle_memory(self) -> None:
        """Toggle memory visualization"""
        self.query_one(EnhancedChatView).toggle_memory_view()
    
    def action_toggle_glyphs(self) -> None:
        """Toggle glyph panel"""
        self.query_one(EnhancedChatView).toggle_glyph_panel()
    
    def action_clear_log(self) -> None:
        """Clear chat log"""
        self.query_one("#chat-log").clear()
    
    def action_show_help(self) -> None:
        """Show help information"""
        self.query_one(EnhancedChatView)._show_help()


if __name__ == "__main__":
    app = EnhancedLuminaApp()
    app.run() 