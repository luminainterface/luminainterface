#!/usr/bin/env python
"""
Test Mistral API Key

This script tests if the Mistral API key in the .env file is valid.
"""

import os
from dotenv import load_dotenv

def test_mistral_api():
    """Test the Mistral API key"""
    # Load API key from .env file
    load_dotenv()
    
    # Get API key
    api_key = os.getenv("MISTRAL_API_KEY")
    
    # Check if key exists
    if not api_key:
        print("No Mistral API key found in .env file.")
        print("Please add MISTRAL_API_KEY=your_key_here to your .env file.")
        return False
    
    # Check if key is valid format
    if len(api_key) < 20:
        print("API key is too short to be valid.")
        return False
    
    # Print masked key
    print(f"Testing Mistral API key: {api_key[:4]}...{api_key[-4:]}")
    
    # Attempt to use the API
    try:
        from mistralai.client import MistralClient
        from mistralai.models.chat_completion import ChatMessage
        
        # Initialize client
        client = MistralClient(api_key=api_key)
        
        # Test listing models
        models = client.list_models()
        print(f"Found {len(models)} models:")
        for model in models:
            print(f"- {model.id}")
        
        # Try to generate a response
        print("\nTrying to generate a response...")
        messages = [ChatMessage(role="user", content="Hello, this is a test.")]
        response = client.chat(model="mistral-medium", messages=messages)
        
        print("\nAPI Response:")
        print(response.choices[0].message.content)
        
        print("\n✅ API key is working correctly!")
        return True
    
    except ImportError:
        print("mistralai package not installed. Install it with: pip install mistralai")
        return False
    except Exception as e:
        print(f"❌ Error testing API key: {str(e)}")
        return False

if __name__ == "__main__":
    print("\n=== Testing Mistral API Key ===\n")
    result = test_mistral_api()
    
    if not result:
        print("\n❌ API key test failed.")
        print("Make sure you have a valid Mistral API key in your .env file.")
        print("Get an API key from: https://console.mistral.ai/")
    else:
        print("\n✅ API key test successful!")