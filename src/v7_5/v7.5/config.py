"""
LUMINA v7.5 Configuration

This module contains configuration settings for the LUMINA v7.5 system.
"""

import os
from pathlib import Path

# Window settings
WINDOW_TITLE = "LUMINA v7.5 - Integrated Neural Interface"
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
MIN_WINDOW_WIDTH = 800
MIN_WINDOW_HEIGHT = 600

# System update interval in ms
SYSTEM_UPDATE_INTERVAL = 5000

# Data directories
DATA_DIR = Path("data")
CONVERSATION_DIR = DATA_DIR / "conversations"
MEMORY_DIR = DATA_DIR / "memory"
LOGS_DIR = Path("logs")

# Ensure directories exist
for directory in [DATA_DIR, CONVERSATION_DIR, MEMORY_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# LLM Settings
LLM_MODEL = "mistral-medium"
LLM_MAX_TOKENS = 1024
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.95
LLM_TOP_K = 50
LLM_FREQUENCY_PENALTY = 0.0
LLM_PRESENCE_PENALTY = 0.0

# Default system prompt
DEFAULT_SYSTEM_PROMPT = """You are LUMINA v7.5, an advanced neural interface system that helps users explore ideas, 
generate insights, and create connections between concepts. You have a neural network structure that allows for
creative thought processes, insight generation, and memory of past conversations. 

Be helpful, creative, and concise in your responses. When possible, draw connections to past conversations and use
the neural network to generate unique insights.
"""

# Neural network settings
NN_WEIGHT = 0.3
LLM_WEIGHT = 0.7
MIN_CONSCIOUSNESS_LEVEL = 0.2

# Conversation settings
CONTEXT_WINDOW_SIZE = 10
MAX_TOPICS = 50
MIN_TOPIC_RELEVANCE = 0.3

# System paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "logs"
MEMORY_DIR = DATA_DIR / "onsite_memory"
CONSCIOUSNESS_DIR = DATA_DIR / "consciousness"
AUTOWIKI_DIR = DATA_DIR / "autowiki"
BREATH_DIR = DATA_DIR / "breath"

# Ensure directories exist
for directory in [DATA_DIR, LOGS_DIR, MEMORY_DIR, CONSCIOUSNESS_DIR, AUTOWIKI_DIR, BREATH_DIR]:
    directory.mkdir(exist_ok=True)

# UI Configuration
WINDOW_WIDTH = 1280  # 16:9 aspect ratio
WINDOW_HEIGHT = 720
MIN_WINDOW_WIDTH = 1000
MIN_WINDOW_HEIGHT = 563  # Maintains 16:9 ratio

# Component settings
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
DASHBOARD_PORT = int(os.environ.get("DASHBOARD_PORT", "8765"))
METRICS_DB_PATH = os.environ.get("METRICS_DB_PATH", str(DATA_DIR / "neural_metrics.db"))

# Feature flags
ENABLE_NODE_CONSCIOUSNESS = os.environ.get("ENABLE_NODE_CONSCIOUSNESS", "true").lower() == "true"
ENABLE_AUTOWIKI = os.environ.get("ENABLE_AUTOWIKI", "true").lower() == "true"
ENABLE_DREAM_MODE = os.environ.get("ENABLE_DREAM_MODE", "true").lower() == "true"
ENABLE_BREATH_DETECTION = os.environ.get("ENABLE_BREATH_DETECTION", "true").lower() == "true"
ENABLE_ONSITE_MEMORY = os.environ.get("ENABLE_ONSITE_MEMORY", "true").lower() == "true"

# Colors
MAIN_BG_COLOR = "#f0f0f0"
TEXT_COLOR = "#303030"
HIGHLIGHT_COLOR = "#4a86e8"
ACCENT_COLOR = "#7c4dff"

# LLM Settings
LLM_PROVIDER = "mistral"
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.95
LLM_TOP_K = 50
LLM_PRESENCE_PENALTY = 0.0
LLM_CONTEXT_WINDOW = 8192
LLM_CACHE_ENABLED = True
LLM_CACHE_TTL = 86400 