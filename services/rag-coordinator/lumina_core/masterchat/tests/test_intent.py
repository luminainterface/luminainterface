import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json
from main import app

client = TestClient(app)

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

@pytest.mark.asyncio
async def test_wiki_intent_detection():
    """Test Wikipedia intent detection"""
    # Test explicit Wikipedia mention
    response = client.post("/masterchat/chat", json={
        "message": "Explain quantum computing from Wikipedia"
    })
    assert response.status_code == 200
    data = response.json()
    assert "quantum" in data["answer"].lower()
    
    # Test question pattern
    response = client.post("/masterchat/chat", json={
        "message": "What is quantum computing?"
    })
    assert response.status_code == 200
    data = response.json()
    assert "quantum" in data["answer"].lower()

@pytest.mark.asyncio
async def test_topic_extraction():
    """Test topic extraction from questions"""
    test_cases = [
        ("Explain quantum computing from Wikipedia", "Quantum Computing"),
        ("What is artificial intelligence?", "Artificial Intelligence"),
        ("Tell me about machine learning", "Machine Learning"),
        ("Describe neural networks", "Neural Networks")
    ]
    
    for question, expected_topic in test_cases:
        response = client.post("/masterchat/chat", json={
            "message": question
        })
        assert response.status_code == 200
        # Topic is used in the crawl, so we should see it in the logs
        assert expected_topic.lower() in response.text.lower()

@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling and retries"""
    with patch('wikipedia.page', side_effect=Exception("API Error")):
        response = client.post("/masterchat/chat", json={
            "message": "Explain quantum computing from Wikipedia"
        })
        assert response.status_code == 500
        assert "error" in response.json()["detail"].lower()

@pytest.mark.asyncio
async def test_metrics():
    """Test Prometheus metrics for Wikipedia queries"""
    # Make a Wikipedia query
    client.post("/masterchat/chat", json={
        "message": "Explain quantum computing from Wikipedia"
    })
    
    # Check metrics endpoint
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "wiki_queries_total" in response.text 