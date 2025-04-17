#!/usr/bin/env python3
"""
Test suite for LUMINA v7.5 Version Transform
"""

import pytest
from datetime import datetime
from typing import Dict, Any

from ..version_transform import MessageTransformer, MessageFormat

@pytest.fixture
def transformer():
    """Setup MessageTransformer instance"""
    return MessageTransformer()

@pytest.mark.asyncio
async def test_message_format_validation(transformer):
    """Test message format validation"""
    # Valid v7.5 message
    valid_message = {
        "type": "test",
        "content": "Hello",
        "timestamp": datetime.now().isoformat(),
        "metadata": {"source": "test"}
    }
    
    format_v75 = transformer.formats["v7.5"]
    valid, error = transformer._validate_message(valid_message, format_v75)
    assert valid
    assert error is None
    
    # Missing required field
    invalid_message = {
        "type": "test",
        "content": "Hello"
        # Missing timestamp
    }
    
    valid, error = transformer._validate_message(invalid_message, format_v75)
    assert not valid
    assert "Missing required fields" in error
    
    # Invalid data type
    invalid_type_message = {
        "type": 123,  # Should be string
        "content": "Hello",
        "timestamp": datetime.now().isoformat()
    }
    
    valid, error = transformer._validate_message(invalid_type_message, format_v75)
    assert not valid
    assert "Invalid type" in error

@pytest.mark.asyncio
async def test_field_mapping(transformer):
    """Test field mapping between versions"""
    # Test direct mapping
    mapping = transformer._get_mapping("v7.5", "v7.0")
    assert mapping is not None
    assert mapping["type"] == "type"
    assert mapping["content"] == "content"
    assert mapping["metadata"] == "metadata"
    
    # Test reverse mapping
    reverse_mapping = transformer._get_mapping("v7.0", "v7.5")
    assert reverse_mapping is not None
    assert reverse_mapping["type"] == "type"
    assert reverse_mapping["content"] == "content"
    
    # Test indirect mapping through intermediate version
    indirect_mapping = transformer._get_mapping("v7.5", "v5.0")
    assert indirect_mapping is not None
    assert "type" in indirect_mapping
    assert "content" in indirect_mapping

@pytest.mark.asyncio
async def test_message_transformation(transformer):
    """Test complete message transformation"""
    # Transform v7.5 to v7.0
    source_message = {
        "type": "test",
        "content": "Hello",
        "timestamp": datetime.now().isoformat(),
        "metadata": {"source": "test"},
        "settings": {"key": "value"}
    }
    
    transformed, error = await transformer.transform_message(
        "v7.5", "v7.0", source_message
    )
    
    assert error is None
    assert transformed is not None
    assert transformed["type"] == source_message["type"]
    assert transformed["content"] == source_message["content"]
    assert transformed["metadata"] == source_message["metadata"]
    
    # Transform v7.0 to v6.0
    v7_message = {
        "type": "test",
        "content": "Hello",
        "metadata": {"source": "test"}
    }
    
    transformed, error = await transformer.transform_message(
        "v7.0", "v6.0", v7_message
    )
    
    assert error is None
    assert transformed is not None
    assert transformed["type"] == v7_message["type"]
    assert transformed["data"] == v7_message["content"]
    assert transformed["meta"] == v7_message["metadata"]

@pytest.mark.asyncio
async def test_error_handling(transformer):
    """Test error handling in transformation"""
    # Test unsupported version
    transformed, error = await transformer.transform_message(
        "unsupported", "v7.0",
        {"type": "test", "content": "Hello"}
    )
    
    assert transformed is None
    assert "Unsupported version" in error
    
    # Test invalid source message
    transformed, error = await transformer.transform_message(
        "v7.5", "v7.0",
        {"invalid": "message"}
    )
    
    assert transformed is None
    assert "Invalid source message" in error
    
    # Test no transformation path
    transformed, error = await transformer.transform_message(
        "v7.5", "v1.0",
        {"type": "test", "content": "Hello"}
    )
    
    assert transformed is None
    assert "No transformation path available" in error

@pytest.mark.asyncio
async def test_version_support(transformer):
    """Test version support checks"""
    supported_versions = transformer.get_supported_versions()
    
    assert "v7.5" in supported_versions
    assert "v7.0" in supported_versions
    assert "v6.0" in supported_versions
    assert "v5.0" in supported_versions
    
    # Verify format definitions
    for version in supported_versions:
        assert version in transformer.formats
        fmt = transformer.formats[version]
        assert isinstance(fmt, MessageFormat)
        assert fmt.version == version
        assert fmt.required_fields
        assert fmt.data_types

@pytest.mark.asyncio
async def test_complex_transformations(transformer):
    """Test complex message transformations"""
    # Test nested content
    nested_message = {
        "type": "complex",
        "content": {
            "data": "test",
            "nested": {
                "field": "value"
            }
        },
        "timestamp": datetime.now().isoformat(),
        "metadata": {"complex": True}
    }
    
    transformed, error = await transformer.transform_message(
        "v7.5", "v7.0", nested_message
    )
    
    assert error is None
    assert transformed is not None
    assert isinstance(transformed["content"], dict)
    assert transformed["content"]["data"] == "test"
    
    # Test array content
    array_message = {
        "type": "array",
        "content": ["item1", "item2"],
        "timestamp": datetime.now().isoformat()
    }
    
    transformed, error = await transformer.transform_message(
        "v7.5", "v6.0", array_message
    )
    
    assert error is None
    assert transformed is not None
    assert isinstance(transformed["data"], list)
    assert len(transformed["data"]) == 2 