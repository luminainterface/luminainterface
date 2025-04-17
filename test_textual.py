from textual.app import App, ComposeResult
from textual.widgets import Header, Static

class SimpleApp(App):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("Hello Lumina!")

if __name__ == "__main__":
    app = SimpleApp()
    app.run() 