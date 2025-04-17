#!/usr/bin/env python
"""
Fix LLM Toggle Issues in Lumina GUI

This script patches potential issues with the LLM integration toggle in Lumina GUI.
"""

import os
import sys
import re
import shutil

def main():
    """Main function to fix LLM toggle issues"""
    print("\n=== Lumina LLM Toggle Fix ===\n")
    
    # Check if lumina_gui.py exists
    if not os.path.exists("lumina_gui.py"):
        print("Error: lumina_gui.py not found.")
        print("Please make sure you're running this script from the project root directory.")
        return
    
    # Create a backup of the file
    backup_path = "lumina_gui.py.bak"
    shutil.copy2("lumina_gui.py", backup_path)
    print(f"Created backup: {backup_path}")
    
    # Read the file content
    with open("lumina_gui.py", "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    # Fix common issues
    fixed_content = fix_toggle_llm_method(content)
    fixed_content = fix_set_llm_enabled_method(fixed_content)
    fixed_content = fix_init_google_client(fixed_content)
    
    # Write the fixed content back to the file
    with open("lumina_gui.py", "w", encoding="utf-8") as f:
        f.write(fixed_content)
    
    print("\nLumina GUI has been patched.")
    print("Please restart the application to apply the changes.")
    print("\nIf the issues persist, you can restore the backup file with:")
    print(f"copy {backup_path} lumina_gui.py")

def fix_toggle_llm_method(content):
    """Fix issues with the toggle_llm method"""
    toggle_llm_pattern = r'def toggle_llm\(self, enabled\):\s+.*?(?=def|\Z)'
    toggle_llm_match = re.search(toggle_llm_pattern, content, re.DOTALL)
    
    if toggle_llm_match:
        current_method = toggle_llm_match.group(0)
        fixed_method = '''def toggle_llm(self, enabled):
        """Toggle LLM integration on/off"""
        try:
            logger.info(f"Setting LLM integration {'enabled' if enabled else 'disabled'}")
            
            # Update the UI
            self.llm_checkbox.setChecked(enabled)
            
            # Update the state
            if hasattr(self.state, 'set_llm_enabled'):
                self.state.set_llm_enabled(enabled)
            else:
                # Direct access if method doesn't exist
                self.state.use_llm = enabled
                
            # Update the status in the chat
            status_message = "LLM integration enabled" if enabled else "LLM integration disabled"
            self.add_system_message(status_message)
            
            # Save the setting to .env file if possible
            try:
                import dotenv
                dotenv_file = dotenv.find_dotenv()
                if dotenv_file:
                    dotenv.set_key(dotenv_file, "ENABLE_LLM_INTEGRATION", str(enabled).lower())
                    logger.info(f"Updated ENABLE_LLM_INTEGRATION in .env to {enabled}")
            except Exception as env_error:
                logger.error(f"Error updating .env file: {str(env_error)}")
                
        except Exception as e:
            logger.error(f"Error toggling LLM integration: {str(e)}")
            
'''
        
        return content.replace(current_method, fixed_method)
    
    return content

def fix_set_llm_enabled_method(content):
    """Fix issues with the set_llm_enabled method in LuminaState"""
    set_llm_enabled_pattern = r'def set_llm_enabled\(self, enabled: bool\):\s+.*?(?=def|\Z)'
    set_llm_enabled_match = re.search(set_llm_enabled_pattern, content, re.DOTALL)
    
    if set_llm_enabled_match:
        current_method = set_llm_enabled_match.group(0)
        fixed_method = '''def set_llm_enabled(self, enabled: bool):
        """Enable or disable LLM integration"""
        logger.info(f"Setting LLM integration {'enabled' if enabled else 'disabled'}")
        self.use_llm = enabled
        
        # If enabling, make sure the LLM client is initialized
        if enabled and hasattr(self, 'llm_integration'):
            try:
                if not self.llm_integration.enabled:
                    logger.info("Re-initializing LLM client...")
                    self.llm_integration._initialize_client()
            except Exception as e:
                logger.error(f"Error re-initializing LLM client: {str(e)}")
        
'''
        
        return content.replace(current_method, fixed_method)
    
    return content

def fix_init_google_client(content):
    """Fix issues with the _init_google_client method"""
    init_google_pattern = r'def _init_google_client\(self\):\s+.*?(?=def|\Z)'
    init_google_match = re.search(init_google_pattern, content, re.DOTALL)
    
    if init_google_match:
        current_method = init_google_match.group(0)
        
        # Check if it has the problematic code
        if 'if not self._check_module_installed' in current_method:
            # Already fixed
            return content
            
        fixed_method = '''def _init_google_client(self):
        """Initialize Google Gemini API client"""
        try:
            # First verify the generativeai module is available
            try:
                import google.generativeai as genai
            except ImportError:
                logger.error("Google generativeai package not found. Install with: pip install google-generativeai")
                self.enabled = False
                return
                
            # Verify API key
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                logger.error("No Google API key found")
                self.enabled = False
                return
                
            # Configure the client
            genai.configure(api_key=api_key)
            self.client = genai
            logger.info(f"Initialized Google Gemini client with model: {self.model}")
            
            # Verify model exists by creating a test model instance
            try:
                model = self.client.GenerativeModel(self.model)
                logger.info(f"Successfully initialized model {self.model}")
            except Exception as model_error:
                logger.error(f"Error initializing model {self.model}: {str(model_error)}")
                logger.info("Falling back to a default model")
                # Try to fall back to a known model
                fallback_models = ["gemini-1.5-flash", "gemini-pro"]
                for fallback in fallback_models:
                    try:
                        logger.info(f"Trying fallback model: {fallback}")
                        self.model = fallback
                        model = self.client.GenerativeModel(self.model)
                        logger.info(f"Successfully initialized fallback model {self.model}")
                        return
                    except:
                        continue
                
                self.enabled = False
                logger.error("Could not initialize any model")
                
        except Exception as e:
            logger.error(f"Error initializing Google client: {str(e)}")
            self.enabled = False
            
'''
        
        return content.replace(current_method, fixed_method)
    
    return content

if __name__ == "__main__":
    main() 