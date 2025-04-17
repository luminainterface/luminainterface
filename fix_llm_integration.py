#!/usr/bin/env python
"""
Fix LLM Integration

This script helps identify and fix common issues with the LLM integration in Lumina.
"""

import os
import sys
import dotenv
import importlib.util
from pathlib import Path

def main():
    """Main function to check and fix LLM integration issues"""
    print("\n=== Lumina LLM Integration Troubleshooter ===\n")
    
    # Check if required files exist
    required_files = [".env", "lumina_gui.py", "lumina_gui_run.py"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"Error: Required files are missing: {', '.join(missing_files)}")
        print("Please make sure you're running this script from the project root directory.")
        return
    
    # Check if required packages are installed
    packages = check_packages()
    if not packages["all_installed"]:
        for package, installed in packages.items():
            if package != "all_installed" and not installed:
                print(f"Missing required package: {package}")
        
        print("\nInstalling missing packages...")
        for package, installed in packages.items():
            if package != "all_installed" and not installed:
                try:
                    import subprocess
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                    print(f"Successfully installed {package}")
                except Exception as e:
                    print(f"Error installing {package}: {str(e)}")
    
    # Load environment variables
    dotenv.load_dotenv()
    
    # Check environment variables
    env_vars = check_env_vars()
    if not env_vars["all_valid"]:
        print("\nFound issues with environment variables:")
        for var, status in env_vars.items():
            if var != "all_valid" and not status:
                print(f"- {var} is invalid or missing")
        
        # Fix environment variables
        fix_env_vars(env_vars)
    
    # Test API key
    print("\nTesting Google Gemini API key...")
    api_key_valid = test_gemini_api()
    
    if not api_key_valid:
        fix_api_key()
    
    # Check if LLM integration is enabled
    llm_enabled = os.getenv("ENABLE_LLM_INTEGRATION", "true").lower() == "true"
    if not llm_enabled:
        print("\nLLM integration is disabled in the .env file.")
        enable = input("Would you like to enable it? (y/n): ").strip().lower()
        if enable == "y":
            dotenv_file = dotenv.find_dotenv()
            dotenv.set_key(dotenv_file, "ENABLE_LLM_INTEGRATION", "true")
            print("LLM integration has been enabled.")
    
    # Final recommendations
    print("\nRecommendations for using LLM integration:")
    print("1. Restart the Lumina application")
    print("2. In the Lumina interface, go to Settings and make sure LLM integration is enabled")
    print("3. Adjust the LLM weight slider to your preference (0.6 is recommended)")
    print("4. Try sending a test message to verify everything is working")
    
    print("\nIf you continue to experience issues, please check the 'lumina.log' file for error messages.")

def check_packages():
    """Check if required packages are installed"""
    packages = {
        "google-generativeai": False,
        "dotenv": False,
        "PyQt5": False,
        "all_installed": False
    }
    
    for package in packages:
        if package != "all_installed":
            packages[package] = importlib.util.find_spec(package.replace("-", ".")) is not None
    
    packages["all_installed"] = all(packages[p] for p in packages if p != "all_installed")
    
    return packages

def check_env_vars():
    """Check if required environment variables are set correctly"""
    env_vars = {
        "GOOGLE_API_KEY": False,
        "LLM_PROVIDER": False,
        "LLM_MODEL": False,
        "ENABLE_LLM_INTEGRATION": False,
        "all_valid": False
    }
    
    # Check API key
    api_key = os.getenv("GOOGLE_API_KEY", "")
    env_vars["GOOGLE_API_KEY"] = api_key and len(api_key) > 10
    
    # Check provider
    provider = os.getenv("LLM_PROVIDER", "")
    env_vars["LLM_PROVIDER"] = provider in ["google", "anthropic", "local", "hybrid"]
    
    # Check model
    model = os.getenv("LLM_MODEL", "")
    env_vars["LLM_MODEL"] = bool(model)
    
    # Check if LLM integration is enabled
    llm_enabled = os.getenv("ENABLE_LLM_INTEGRATION", "").lower()
    env_vars["ENABLE_LLM_INTEGRATION"] = llm_enabled in ["true", "false"]
    
    # Check if all variables are valid
    env_vars["all_valid"] = all(env_vars[v] for v in env_vars if v != "all_valid")
    
    return env_vars

def fix_env_vars(env_vars):
    """Fix invalid environment variables"""
    dotenv_file = dotenv.find_dotenv()
    
    # Fix provider
    if not env_vars["LLM_PROVIDER"]:
        dotenv.set_key(dotenv_file, "LLM_PROVIDER", "google")
        print("Set LLM_PROVIDER to 'google'")
    
    # Fix model
    if not env_vars["LLM_MODEL"]:
        dotenv.set_key(dotenv_file, "LLM_MODEL", "gemini-1.5-flash")
        print("Set LLM_MODEL to 'gemini-1.5-flash'")
    
    # Fix enable flag
    if not env_vars["ENABLE_LLM_INTEGRATION"]:
        dotenv.set_key(dotenv_file, "ENABLE_LLM_INTEGRATION", "true")
        print("Set ENABLE_LLM_INTEGRATION to 'true'")

def fix_api_key():
    """Guide the user to fix their API key"""
    print("\nYou need a valid Google Gemini API key for LLM integration to work.")
    print("To get a key, go to: https://aistudio.google.com/app/apikey")
    
    open_browser = input("Would you like me to open the API key page in your browser? (y/n): ").strip().lower()
    if open_browser == "y":
        import webbrowser
        webbrowser.open("https://aistudio.google.com/app/apikey")
    
    new_key = input("\nEnter your new Google Gemini API key: ").strip()
    if new_key:
        dotenv_file = dotenv.find_dotenv()
        dotenv.set_key(dotenv_file, "GOOGLE_API_KEY", new_key)
        print("API key has been updated.")
        
        # Test the new key
        test_gemini_api()

def test_gemini_api():
    """Test if the Google Gemini API key is valid"""
    try:
        # Check if the package is installed
        import google.generativeai as genai
        
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            print("No API key found")
            return False
        
        # Configure the client
        genai.configure(api_key=api_key)
        
        # Try to list models
        try:
            models = genai.list_models()
            print(f"Found {len(models)} available models")
            
            # Check if our model is in the list
            model_name = os.getenv("LLM_MODEL", "gemini-1.5-flash")
            model_available = any(model_name in model.name for model in models)
            
            if not model_available:
                print(f"Warning: Model '{model_name}' not found in available models")
                available_models = [model.name for model in models if "gemini" in model.name]
                if available_models:
                    print(f"Available Gemini models: {', '.join(available_models[:5])}...")
                    # Update the model in .env
                    if available_models:
                        dotenv_file = dotenv.find_dotenv()
                        dotenv.set_key(dotenv_file, "LLM_MODEL", available_models[0])
                        print(f"Updated LLM_MODEL to {available_models[0]}")
            
            # Test a simple generation
            try:
                test_model = available_models[0] if available_models else "gemini-1.5-flash"
                model = genai.GenerativeModel(test_model)
                response = model.generate_content("Say 'Hello, I am working!' if you can read this message.")
                
                if hasattr(response, 'text'):
                    print(f"API test successful: {response.text}")
                    return True
                else:
                    print("API test failed: Unexpected response format")
                    return False
            except Exception as e:
                print(f"Error testing model generation: {str(e)}")
                return False
                
        except Exception as e:
            print(f"Error listing models: {str(e)}")
            return False
            
    except ImportError:
        print("Google generativeai package not installed")
        return False
    except Exception as e:
        print(f"API test failed: {str(e)}")
        return False

if __name__ == "__main__":
    main() 