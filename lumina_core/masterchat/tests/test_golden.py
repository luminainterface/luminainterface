import pytest
import json
import os
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from main import app

client = TestClient(app)

def load_golden_tests():
    """Load all golden test cases"""
    golden_dir = Path(__file__).parent / "golden" / "wiki_answers"
    test_cases = []
    
    for json_file in golden_dir.glob("*.json"):
        with open(json_file) as f:
            test_case = json.load(f)
            test_cases.append((
                test_case["question"],
                test_case["expected"],
                json_file.name
            ))
    
    return test_cases

@pytest.fixture
def mock_wikipedia():
    """Mock Wikipedia API responses"""
    with patch('wikipedia.search') as mock_search, \
         patch('wikipedia.page') as mock_page:
        
        # Mock search results
        mock_search.return_value = [
            "Quantum Computing",
            "Quantum Computer",
            "Quantum Information"
        ]
        
        # Mock page content
        mock_page.return_value = MagicMock(
            title="Quantum Computing",
            url="https://en.wikipedia.org/wiki/Quantum_Computing",
            content="Quantum computing is a type of computing...",
            summary="Quantum computing is a type of computing..."
        )
        
        yield mock_search, mock_page

@pytest.fixture
def mock_mistral():
    """Mock Mistral API responses"""
    with patch('llm_client.chat') as mock_chat:
        # Mock summarization response
        mock_chat.side_effect = [
            MagicMock(choices=[MagicMock(message=MagicMock(content=json.dumps({
                "summary": "Quantum computing is a revolutionary technology...",
                "facts": [
                    "Uses quantum bits (qubits)",
                    "Can solve certain problems exponentially faster",
                    "Based on quantum mechanics principles"
                ]
            })))]),
            # Mock QA response
            MagicMock(choices=[MagicMock(message=MagicMock(content=json.dumps({
                "answer": "Quantum computing is a type of computing that uses quantum bits...",
                "confidence": 0.95,
                "sources": ["Quantum Computing", "Quantum Computer"]
            })))])
        ]
        yield mock_chat

@pytest.mark.parametrize("question,expected,test_file", load_golden_tests())
@pytest.mark.asyncio
async def test_golden_answers(question, expected, test_file, mock_wikipedia, mock_mistral):
    """Test that answers match golden test cases"""
    # Make request to chat endpoint
    response = client.post("/masterchat/chat", json={
        "message": question
    })
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    
    # Compare with golden test
    assert "answer" in data
    assert "confidence" in data
    assert "sources" in data
    assert "facts" in data
    
    # Check answer content
    assert expected["answer"].lower() in data["answer"].lower()
    
    # Check confidence is within 0.1 of expected
    assert abs(data["confidence"] - expected["confidence"]) <= 0.1
    
    # Check sources
    for source in expected["sources"]:
        assert any(source.lower() in s.lower() for s in data["sources"])
    
    # Check facts
    for fact in expected["facts"]:
        assert any(fact.lower() in f.lower() for f in data["facts"])

def test_golden_files_exist():
    """Test that all golden test files exist and are valid JSON"""
    golden_dir = Path(__file__).parent / "golden" / "wiki_answers"
    assert golden_dir.exists(), "Golden test directory not found"
    
    json_files = list(golden_dir.glob("*.json"))
    assert len(json_files) > 0, "No golden test files found"
    
    for json_file in json_files:
        with open(json_file) as f:
            try:
                test_case = json.load(f)
                assert "question" in test_case
                assert "expected" in test_case
                assert "answer" in test_case["expected"]
                assert "confidence" in test_case["expected"]
                assert "sources" in test_case["expected"]
                assert "facts" in test_case["expected"]
            except json.JSONDecodeError:
                pytest.fail(f"Invalid JSON in {json_file}")
            except AssertionError:
                pytest.fail(f"Missing required fields in {json_file}") 