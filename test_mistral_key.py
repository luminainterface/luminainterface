#!/usr/bin/env python3
"""
Simple script to test Mistral API key
"""

import os
import sys
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# Set your API key here or use environment variable
API_KEY = "nLKZEpq29OihnaArxV7s6KtzsNEiky2A"
# Alternative formats to try
API_KEY_FORMATS = [
    API_KEY,
    f"mistr-{API_KEY}",
    f"mistral-{API_KEY}"
]

def test_key(key):
    """Test if a Mistral API key works"""
    print(f"Testing API key: {key[:4]}...{key[-4:]} (length: {len(key)})")
    
    try:
        # Initialize the client
        client = MistralClient(api_key=key)
        
        # Test with a simple chat completion
        messages = [
            ChatMessage(role="user", content="Hello, how are you?")
        ]
        
        # Make the API call
        response = client.chat(
            model="mistral-small",
            messages=messages,
        )
        
        # Print the response
        print("API call successful!")
        print(f"Response: {response.choices[0].message.content[:100]}...")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Main function"""
    print("Mistral API Key Tester")
    print("=====================")
    
    # Try each format
    for i, key_format in enumerate(API_KEY_FORMATS):
        print(f"\nTest {i+1}: Testing key format")
        success = test_key(key_format)
        if success:
            print(f"SUCCESS! Key format works: {key_format[:4]}...{key_format[-4:]}")
            return 0
    
    print("\nAll key formats failed. The API key may be invalid.")
    return 1

if __name__ == "__main__":
    sys.exit(main()) 