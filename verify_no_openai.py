#!/usr/bin/env python
"""
Verify that OpenAI integration is disabled in Lumina
"""

import os
import sys
import dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LuminaVerifier")

def check_env_variables():
    """Check environment variables for OpenAI settings"""
    logger.info("Checking environment variables for OpenAI settings...")
    
    # Load environment variables
    dotenv.load_dotenv()
    
    openai_related = {}
    provider = os.getenv("LLM_PROVIDER", "google")
    openai_related["LLM_PROVIDER"] = provider
    
    openai_key = os.getenv("OPENAI_API_KEY", None)
    openai_related["OPENAI_API_KEY"] = "Present" if openai_key else "Not set"
    
    openai_model = os.getenv("OPENAI_MODEL", None)
    openai_related["OPENAI_MODEL"] = openai_model if openai_model else "Not set"
    
    # Print the results
    logger.info("OpenAI-related settings:")
    for key, value in openai_related.items():
        logger.info(f"- {key}: {value}")
    
    # Verify OpenAI is not used
    if provider == "openai":
        logger.error("LLM_PROVIDER is set to 'openai'. This should be changed to another provider like 'google', 'anthropic', or 'local'.")
        return False
    
    if openai_key and openai_key != "your_openai_key_here":
        logger.warning("OPENAI_API_KEY is set. While not currently used, it's recommended to comment this out or remove it.")
    else:
        logger.info("No active OpenAI API key found.")
    
    if provider != "openai":
        logger.info(f"LLM_PROVIDER is set to '{provider}', which is not OpenAI.")
    
    return provider != "openai"

def check_import_statements():
    """Check if openai package is imported in key files"""
    logger.info("Checking import statements for OpenAI references...")
    
    files_to_check = ["lumina_gui.py"]
    openai_imports = []
    
    for file in files_to_check:
        if not os.path.exists(file):
            logger.warning(f"File '{file}' not found, skipping import check")
            continue
        
        try:
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if "import openai" in content or "from openai import" in content:
                    openai_imports.append(file)
        except Exception as e:
            logger.error(f"Error reading file {file}: {str(e)}")
    
    if openai_imports:
        logger.warning(f"Found imports of OpenAI in the following files: {', '.join(openai_imports)}")
        logger.warning("These imports may not be used but should be reviewed.")
        return False
    else:
        logger.info("No OpenAI imports found in critical files.")
        return True

def check_openai_client_initialization():
    """Check if OpenAI client is initialized in the code"""
    logger.info("Checking for OpenAI client initialization...")
    
    files_to_check = ["lumina_gui.py"]
    openai_init = []
    
    for file in files_to_check:
        if not os.path.exists(file):
            logger.warning(f"File '{file}' not found, skipping client init check")
            continue
        
        try:
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if "openai.api_key" in content:
                    openai_init.append(file)
        except Exception as e:
            logger.error(f"Error reading file {file}: {str(e)}")
    
    if openai_init:
        logger.warning(f"Found OpenAI client initialization in the following files: {', '.join(openai_init)}")
        logger.warning("The OpenAI client should not be initialized anywhere in the code.")
        return False
    else:
        logger.info("No OpenAI client initialization found.")
        return True

def main():
    """Main verification function"""
    logger.info("Starting verification of OpenAI integration status...")
    
    checks = [
        check_env_variables(),
        check_import_statements(),
        check_openai_client_initialization()
    ]
    
    if all(checks):
        logger.info("All checks passed! OpenAI integration is disabled.")
        return 0
    else:
        logger.warning("Some checks failed. OpenAI integration may not be fully disabled.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 