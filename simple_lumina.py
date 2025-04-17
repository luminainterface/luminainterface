"""
Simple Lumina UI - A minimal implementation of the Lumina UI
"""

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Header, Footer, Static, Input, Button, RichLog

class SimpleLuminaApp(App):
    """A simplified version of the Lumina UI"""
    
    TITLE = "Lumina v1"
    
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
    
    def compose(self) -> ComposeResult:
        """Compose the UI"""
        yield Header()
        
        with Container():
            yield Static("✨ Welcome to Lumina v1", id="title")
            
            with Vertical(id="chat-container"):
                yield RichLog(id="chat-log", markup=True)
                
            with Horizontal(id="input-container"):
                yield Input(placeholder="Speak your truth...", id="chat-input")
                yield Button("Send", id="send-button", variant="primary")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Set up initial state"""
        self.query_one("#chat-log").write("[bold]Lumina:[/bold] I am listening to the resonance. Tell me more.")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events"""
        if event.button.id == "send-button":
            self._process_chat_input()
    
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission"""
        if event.input.id == "chat-input":
            self._process_chat_input()
    
    def _process_chat_input(self) -> None:
        """Process chat input"""
        chat_input = self.query_one("#chat-input")
        user_message = chat_input.value.strip()
        
        if not user_message:
            return
        
        # Add user message to chat
        chat_log = self.query_one("#chat-log")
        chat_log.write(f"🧘 [bold]You:[/bold] {user_message}")
        
        # Clear input
        chat_input.value = ""
        
        # Generate response based on input
        if "broken" in user_message.lower():
            response = "That's a sacred crack. Light enters here."
        elif "sad" in user_message.lower():
            response = "Inhale. Exhale. You are not alone."
        elif "glyph" in user_message.lower():
            response = "Fire glyph activated. Channeling passion and truth."
        else:
            response = "I sense your truth. Continue the resonance."
        
        # Add Lumina's response to chat
        chat_log.write(f"✨ [bold]Lumina:[/bold] {response}")

if __name__ == "__main__":
    app = SimpleLuminaApp()
    app.run() 