#!/usr/bin/env python
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import httpx
import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text

# Add parent directory to path for local imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from lumina_core.masterchat.main import chat_sync  # local direct call

app = typer.Typer(help="MasterChat CLI - Ask questions and get AI-powered answers")
console = Console()

# Set up history and config files
HISTORY_FILE = Path.home() / ".masterchat_history"
CONFIG_FILE = Path.home() / ".masterchat_cfg.json"
HISTORY_FILE.parent.mkdir(exist_ok=True)
HISTORY_FILE.touch(exist_ok=True)

DEFAULT_CONFIG = {
    "url": "http://localhost:8000",
    "api_key": "",
    "default_stream": False,
    "default_local": False
}

def load_config() -> Dict[str, Any]:
    """Load configuration from file"""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                return {**DEFAULT_CONFIG, **json.load(f)}
        except json.JSONDecodeError:
            console.print("[yellow]Warning:[/yellow] Invalid config file, using defaults")
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()

def save_config(config: Dict[str, Any]):
    """Save configuration to file"""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

def build_agent_table(steps):
    """Build a table showing agent execution status"""
    table = Table(show_header=False, box=None, padding=(0, 1))
    for s in steps:
        style = {
            "pending": "dim",
            "start": "yellow",
            "end": "bold green",
            "error": "bold red"
        }.get(s["status"], "white")
        
        table.add_column(style=style, justify="center", no_wrap=True)
        table.add_row(s["agent"][:3])
    return table

def build_progress_panel(message: str, progress: Progress):
    """Build a panel showing current progress"""
    return Panel(
        progress,
        title="Processing",
        subtitle=message,
        border_style="blue"
    )

def interactive_mode(url: str, api_key: str, local: bool, stream: bool):
    """Run in interactive mode with history"""
    session = PromptSession(history=FileHistory(str(HISTORY_FILE)))
    
    console.print("[bold blue]MasterChat Interactive Mode[/bold blue]")
    console.print("Type your questions and press Enter. Press Ctrl+D or Ctrl+C to exit.")
    console.print()
    
    while True:
        try:
            question = session.prompt("Question: ")
            if not question.strip():
                continue
                
            # Run the question
            asyncio.run(process_question(question, url, api_key, local, stream))
            console.print()
            
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye!")
            break
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {str(e)}")

async def process_question(question: str, url: str, api_key: str, local: bool, stream: bool):
    """Process a single question"""
    # Initial agent steps
    steps = [
        {"agent": "CrawlAgent", "status": "pending"},
        {"agent": "SummariseAgent", "status": "pending"},
        {"agent": "QAAgent", "status": "pending"},
    ]
    
    with Live(
        build_agent_table(steps),
        console=console,
        refresh_per_second=4
    ) as live:
        try:
            if local:
                # Run in-process
                answer = await chat_sync(question)
                console.print("\n[bold]Answer:[/bold]", answer)
            else:
                # Run via HTTP
                async with httpx.AsyncClient(timeout=None) as client:
                    if stream:
                        # Handle streaming response
                        async with client.stream(
                            "POST",
                            f"{url}/masterchat/chat",
                            headers={"X-API-Key": api_key} if api_key else {},
                            json={"message": question},
                            timeout=None
                        ) as response:
                            response.raise_for_status()
                            async for line in response.aiter_lines():
                                if line.startswith("data: "):
                                    try:
                                        data = json.loads(line[6:])
                                        if "delta" in data:
                                            console.print(data["delta"].get("content", ""), end="")
                                        elif "answer" in data:
                                            console.print("\n[bold]Answer:[/bold]", data["answer"])
                                    except json.JSONDecodeError:
                                        continue
                    else:
                        # Handle regular response
                        response = await client.post(
                            f"{url}/masterchat/chat",
                            headers={"X-API-Key": api_key} if api_key else {},
                            json={"message": question},
                            timeout=None
                        )
                        response.raise_for_status()
                        data = response.json()
                        console.print("\n[bold]Answer:[/bold]", data["answer"])
                        
                        if "sources" in data:
                            console.print("\n[bold]Sources:[/bold]")
                            for source in data["sources"]:
                                console.print(f"• {source}")
                        
                        if "facts" in data:
                            console.print("\n[bold]Key Facts:[/bold]")
                            for fact in data["facts"]:
                                console.print(f"• {fact}")
            
        except httpx.HTTPError as e:
            console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
            sys.exit(1)
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
            sys.exit(1)

@app.command()
def config(
    show: bool = typer.Option(False, "--show", help="Show current configuration"),
    set_url: Optional[str] = typer.Option(None, help="Set API URL"),
    set_key: Optional[str] = typer.Option(None, help="Set API key"),
    set_stream: Optional[bool] = typer.Option(None, help="Set default streaming mode"),
    set_local: Optional[bool] = typer.Option(None, help="Set default local mode"),
    reset: bool = typer.Option(False, help="Reset to default configuration")
):
    """
    Manage MasterChat CLI configuration.
    
    Example:
        masterchat config --show
        masterchat config --set-url https://api.prod
        masterchat config --set-key sk-123
    """
    config = load_config()
    
    if reset:
        config = DEFAULT_CONFIG.copy()
        save_config(config)
        console.print("[green]Configuration reset to defaults[/green]")
        return
    
    if show:
        table = Table(title="Current Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in config.items():
            if key == "api_key" and value:
                value = "********"  # Mask API key
            table.add_row(key, str(value))
        
        console.print(table)
        return
    
    # Update configuration
    if set_url is not None:
        config["url"] = set_url
    if set_key is not None:
        config["api_key"] = set_key
    if set_stream is not None:
        config["default_stream"] = set_stream
    if set_local is not None:
        config["default_local"] = set_local
    
    save_config(config)
    console.print("[green]Configuration updated[/green]")

@app.command()
def ask(
    question: Optional[str] = typer.Argument(None, help="The question to ask"),
    url: str = typer.Option(None, help="MasterChat API URL"),
    api_key: str = typer.Option(None, help="API key for authentication"),
    local: bool = typer.Option(None, help="Run agents in-process instead of via HTTP"),
    stream: bool = typer.Option(None, help="Show streaming response (SSE)"),
    interactive: bool = typer.Option(
        False,
        "--interactive", "-i",
        help="Run in interactive mode with command history"
    )
):
    """
    Ask MasterChat a question and get an AI-powered answer.
    
    Example:
        masterchat ask "Who is Alan Turing from Wikipedia?"
    """
    # Load configuration
    config = load_config()
    
    # Use command line args if provided, otherwise use config
    url = url or config["url"]
    api_key = api_key or config["api_key"]
    local = local if local is not None else config["default_local"]
    stream = stream if stream is not None else config["default_stream"]
    
    if interactive:
        interactive_mode(url, api_key, local, stream)
    elif question:
        asyncio.run(process_question(question, url, api_key, local, stream))
    else:
        console.print("[bold red]Error:[/bold red] Please provide a question or use --interactive mode")
        sys.exit(1)

if __name__ == "__main__":
    app() 