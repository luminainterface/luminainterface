#!/usr/bin/env python
"""
Verify LLM Integration in Lumina

This script tests if the LLM integration is working properly in Lumina.
"""

import os
import sys
import time
import dotenv

def main():
    """Main function to verify LLM integration"""
    print("\n=== Verifying Lumina LLM Integration ===\n")
    
    # Check if required modules are available
    try:
        import mistralai.client
        import PyQt5
        print("✅ Required packages are installed")
    except ImportError as e:
        print(f"❌ Missing package: {str(e)}")
        print("Please run 'pip install mistralai PyQt5' to install required packages")
        return
    
    # Check environment variables
    dotenv.load_dotenv()
    
    # Check API key
    api_key = os.getenv("MISTRAL_API_KEY", "")
    if not api_key or len(api_key) < 10:
        print("❌ API key is missing or invalid")
        print("Please get a valid API key from https://console.mistral.ai/")
        return
    
    # Check LLM settings
    llm_provider = os.getenv("LLM_PROVIDER", "")
    llm_model = os.getenv("LLM_MODEL", "")
    llm_enabled = os.getenv("ENABLE_LLM_INTEGRATION", "").lower() == "true"
    
    if llm_provider != "mistral":
        print(f"❌ LLM_PROVIDER is set to '{llm_provider}' instead of 'mistral'")
    else:
        print("✅ LLM_PROVIDER is set to 'mistral'")
    
    if not llm_model:
        print("❌ LLM_MODEL is not set")
    else:
        print(f"✅ LLM_MODEL is set to '{llm_model}'")
    
    if not llm_enabled:
        print("❌ ENABLE_LLM_INTEGRATION is not set to 'true'")
    else:
        print("✅ ENABLE_LLM_INTEGRATION is set to 'true'")
    
    # Test API key with Mistral AI
    print("\nTesting API key with Mistral AI...")
    try:
        from mistralai.client import MistralClient
        from mistralai.models.chat_completion import ChatMessage
        
        # Initialize client
        mistral_client = MistralClient(api_key=api_key)
        
        # Test listing models
        try:
            models = mistral_client.list_models()
            model_names = [model.id for model in models]
            print(f"✅ API key is valid. Found {len(models)} models")
            
            # Check if the configured model is available
            if llm_model and any(llm_model == name for name in model_names):
                print(f"✅ Configured model '{llm_model}' is available")
            else:
                print(f"❌ Configured model '{llm_model}' not found in available models")
                print(f"Available Mistral models: {', '.join(model_names)}")
                print(f"Consider updating your .env file to use one of these models")
            
            # Test generating content
            test_model = llm_model if llm_model and any(llm_model == name for name in model_names) else "mistral-medium"
            print(f"\nTesting content generation with model '{test_model}'...")
            
            messages = [ChatMessage(role="user", content="Hello, can you respond with a short greeting?")]
            response = mistral_client.chat(model=test_model, messages=messages)
            
            if response and response.choices and response.choices[0].message.content:
                print(f"✅ Successfully generated content: '{response.choices[0].message.content[:50]}...'")
            else:
                print("❌ Failed to generate content: Response format is unexpected")
            
        except Exception as e:
            print(f"❌ Error listing models: {str(e)}")
    except Exception as e:
        print(f"❌ Error testing API key: {str(e)}")
    
    # Read directly from .env file
    print("\nReading .env file directly:")
    try:
        with open(".env", "r", encoding="utf-8") as f:
            env_contents = f.read()
            print("\n.env file contents:")
            print(env_contents[:500] + "..." if len(env_contents) > 500 else env_contents)
    except Exception as e:
        print(f"Error reading .env file: {str(e)}")
    
    # Final verification and instructions
    if all([
        api_key and len(api_key) >= 10,
        llm_provider == "mistral",
        llm_model,
        llm_enabled
    ]):
        print("\n✅ All checks passed! LLM integration should be working properly.")
        print("\nTo use LLM integration in Lumina:")
        print("1. Start Lumina: python lumina_gui_run.py")
        print("2. In the Lumina interface, make sure 'Enable LLM Integration' is checked")
        print("3. Adjust the LLM weight as needed (0.6 is recommended)")
    else:
        print("\n❌ Some checks failed. LLM integration may not work properly.")
        print("Please run 'python fix_llm_integration.py' to fix common issues.")
    
    print("\nIf you still encounter issues, check the 'lumina.log' file for error messages.")

if __name__ == "__main__":
    main() 