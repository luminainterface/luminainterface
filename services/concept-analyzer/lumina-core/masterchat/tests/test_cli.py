import pytest
import subprocess
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

@pytest.fixture
def cli_path():
    """Get the path to the CLI script"""
    return Path(__file__).parent.parent / "scripts" / "masterchat_cli.py"

@pytest.fixture
def config_file():
    """Get the path to the config file"""
    return Path.home() / ".masterchat_cfg.json"

def test_cli_help(cli_path):
    """Test that the CLI help command works"""
    result = subprocess.run(
        ["python", str(cli_path), "--help"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "MasterChat CLI" in result.stdout
    assert "Ask MasterChat a question" in result.stdout

def test_config_command(cli_path, config_file):
    """Test config command functionality"""
    # Test showing config
    result = subprocess.run(
        ["python", str(cli_path), "config", "--show"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "Current Configuration" in result.stdout
    assert "url" in result.stdout
    assert "api_key" in result.stdout
    
    # Test setting URL
    result = subprocess.run(
        ["python", str(cli_path), "config", "--set-url", "https://api.test"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "Configuration updated" in result.stdout
    
    # Verify config file
    with open(config_file) as f:
        config = json.load(f)
    assert config["url"] == "https://api.test"
    
    # Test setting API key
    result = subprocess.run(
        ["python", str(cli_path), "config", "--set-key", "test-key"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    
    # Verify API key is masked in show
    result = subprocess.run(
        ["python", str(cli_path), "config", "--show"],
        capture_output=True,
        text=True
    )
    assert "********" in result.stdout
    
    # Test reset
    result = subprocess.run(
        ["python", str(cli_path), "config", "--reset"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "reset to defaults" in result.stdout

@pytest.mark.asyncio
async def test_cli_local_mode(cli_path):
    """Test CLI in local mode with a golden test question"""
    # Mock the chat_sync function
    with patch("lumina_core.masterchat.main.chat_sync") as mock_chat:
        mock_chat.return_value = {
            "answer": "Quantum computing is a type of computing that uses quantum bits...",
            "confidence": 0.95,
            "sources": ["Quantum Computing", "Quantum Computer"],
            "facts": [
                "Uses quantum bits (qubits)",
                "Can solve certain problems exponentially faster"
            ]
        }
        
        # Run the CLI in local mode
        result = subprocess.run(
            ["python", str(cli_path), "ask", "What is quantum computing?", "--local"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "Quantum computing" in result.stdout
        assert "Sources:" in result.stdout
        assert "Key Facts:" in result.stdout

@pytest.mark.asyncio
async def test_cli_http_mode(cli_path):
    """Test CLI in HTTP mode with mocked response"""
    # Mock httpx.AsyncClient
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "answer": "Artificial Intelligence is the simulation of human intelligence...",
            "confidence": 0.92,
            "sources": ["Artificial Intelligence", "Machine Learning"],
            "facts": [
                "Simulates human intelligence in machines",
                "Includes machine learning and deep learning"
            ]
        }
        mock_post.return_value = mock_response
        
        # Run the CLI in HTTP mode
        result = subprocess.run(
            ["python", str(cli_path), "ask", "What is AI?"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "Artificial Intelligence" in result.stdout
        assert "Sources:" in result.stdout
        assert "Key Facts:" in result.stdout

@pytest.mark.asyncio
async def test_cli_streaming_mode(cli_path):
    """Test CLI in streaming mode"""
    # Mock httpx.AsyncClient.stream
    with patch("httpx.AsyncClient.stream") as mock_stream:
        mock_response = MagicMock()
        mock_response.aiter_lines.return_value = [
            "data: {\"delta\": {\"content\": \"Artificial\"}}",
            "data: {\"delta\": {\"content\": \" Intelligence\"}}",
            "data: {\"answer\": \"Artificial Intelligence is...\"}"
        ]
        mock_stream.return_value.__aenter__.return_value = mock_response
        
        # Run the CLI in streaming mode
        result = subprocess.run(
            ["python", str(cli_path), "ask", "What is AI?", "--stream"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "Artificial Intelligence" in result.stdout

def test_cli_error_handling(cli_path):
    """Test CLI error handling"""
    # Test with invalid URL
    result = subprocess.run(
        ["python", str(cli_path), "ask", "What is AI?", "--url", "http://invalid"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 1
    assert "Error:" in result.stdout 