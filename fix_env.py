#!/usr/bin/env python
"""
Fix .env file for Lumina LLM integration
"""

def main():
    """Generate a clean .env file with correct LLM settings"""
    env_content = """# LLM Settings
LLM_PROVIDER=google
LLM_MODEL=models/gemini-1.5-pro-latest
LLM_MAX_TOKENS=1024
LLM_TEMPERATURE=0.7
LLM_TOP_P=0.95
LLM_PRESENCE_PENALTY=0.0
LLM_FREQUENCY_PENALTY=0.0
LLM_CONTEXT_WINDOW=8192
LLM_CACHE_ENABLED=true
LLM_CACHE_TTL=86400

# Google API Configuration
GOOGLE_API_KEY=AIzaSyD5xa2pBrvXUYoAUCDRLm__yRlJVg46Y2A
GOOGLE_CLOUD_PROJECT=neural-networkl
GOOGLE_CLOUD_LOCATION=us-central1

# Feature Flags
ENABLE_BACKGROUND_LEARNING=true
ENABLE_HYBRID_NODE=true
ENABLE_WIKI_INTEGRATION=true
ENABLE_QUANTUM_INFECTION=true
ENABLE_TRAINING_DATA_COLLECTION=true
ENABLE_FALLBACK_RESPONSES=true
ENABLE_LLM_INTEGRATION=true
ENABLE_WEIGHT_ADJUSTMENT_UI=true
ENABLE_RESPONSE_BLENDING=true

# Weights
DEFAULT_LLM_WEIGHT=0.6
DEFAULT_NN_WEIGHT=0.4
MIN_LLM_WEIGHT=0.0
MAX_LLM_WEIGHT=1.0
WEIGHT_STEP=0.05

# Neural Network Settings
USE_LOCAL_MODELS=true
MODEL_PATH=models
DEVICE=cpu
"""

    try:
        with open('.env', 'w', encoding='utf-8') as f:
            f.write(env_content)
        print("Successfully created clean .env file with correct LLM settings")
    except Exception as e:
        print(f"Error writing .env file: {str(e)}")

if __name__ == "__main__":
    main() 