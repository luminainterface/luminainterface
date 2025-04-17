# Enhanced Language System

A sophisticated language processing system that integrates consciousness, neural linguistics, language memory, and recursive pattern analysis with LLM weighing capabilities and Mistral API integration.

## Overview

The Enhanced Language System is a Python-based framework that implements advanced language processing capabilities by integrating multiple specialized components:

1. **Language Memory** - Stores and recalls associations between words and sentences
2. **Conscious Mirror Language** - Analyzes consciousness levels in text
3. **Neural Linguistic Processor** - Identifies patterns and semantic relationships
4. **Recursive Pattern Analyzer** - Detects self-references and linguistic loops
5. **Central Language Node** - Orchestrates all components with unified LLM weighing
6. **Database Manager** - Provides persistent storage for conversations and learning
7. **Conversation Memory** - Tracks interaction history with concept extraction
8. **Mistral Integration** - Connects to Mistral AI API for enhanced responses
9. **Language Database Bridge** - Ensures database synchronization across components

Each component can be used independently or as part of the integrated system through the Central Language Node.

## Core Capabilities

- **Association-based language memory** with recall functionality
- **Consciousness level detection** in language processing
- **Neural linguistic pattern recognition** for semantic analysis
- **Self-reference and recursive pattern detection** in text
- **LLM weight adjustment** across all components
- **Cross-domain mappings** between language, consciousness, and neural patterns
- **Persistent conversation storage** for learning from interactions
- **Mistral AI model integration** for enhanced response generation
- **Bidirectional database synchronization** for reliable data persistence

## Installation

```bash
# Clone the repository
git clone https://github.com/username/neural_network_project.git
cd neural_network_project

# Install dependencies
pip install -r requirements.txt
pip install mistralai
```

## Directory Structure

```
neural_network_project/
├── data/                     # Data storage for all components
│   ├── memory/               # Language memory storage
│   ├── v10/                  # Conscious Mirror Language data
│   ├── neural_linguistic/    # Neural linguistic data
│   ├── recursive_patterns/   # Pattern analysis data
│   ├── db/                   # Database storage
│   ├── conversation_memory/  # Conversation memory storage
│   └── logs/                 # Log files
├── docs/                     # Documentation
├── src/                      # Source code
│   ├── language/             # Language system components
│   │   ├── language_memory.py
│   │   ├── conscious_mirror_language.py
│   │   ├── neural_linguistic_processor.py
│   │   ├── recursive_pattern_analyzer.py
│   │   ├── central_language_node.py
│   │   ├── database_manager.py
│   │   ├── conversation_memory.py
│   │   └── language_database_bridge.py
│   ├── chat_with_system.py   # Interactive chat interface
│   ├── mistral_integration.py # Mistral API integration
│   └── verify_database_connections.py # Database connection verification
└── tests/                    # Test scripts
```

## Usage Examples

### Basic Component Usage

```python
from language.language_memory import LanguageMemory
from language.conscious_mirror_language import ConsciousMirrorLanguage
from language.neural_linguistic_processor import NeuralLinguisticProcessor
from language.recursive_pattern_analyzer import RecursivePatternAnalyzer

# Initialize Language Memory
memory = LanguageMemory(data_dir="data/memory/language_memory", llm_weight=0.5)
memory.store_word_association("neural", "network", 0.8)
associations = memory.recall_associations("neural")

# Initialize Conscious Mirror Language
cml = ConsciousMirrorLanguage(data_dir="data/v10", llm_weight=0.5)
result = cml.process_text("The system becomes conscious of its own language.")

# Initialize Neural Linguistic Processor
nlp = NeuralLinguisticProcessor(data_dir="data/neural_linguistic", llm_weight=0.5)
result = nlp.process_text("Neural networks process language patterns.")

# Initialize Recursive Pattern Analyzer
analyzer = RecursivePatternAnalyzer(data_dir="data/recursive_patterns", llm_weight=0.5)
result = analyzer.analyze_text("This sentence refers to itself and contains patterns.")
```

### Integrated System Usage

```python
from language.central_language_node import CentralLanguageNode

# Initialize the Central Language Node
central_node = CentralLanguageNode(data_dir="data", llm_weight=0.5)

# Process text through the integrated system
result = central_node.process_text("The neural system integrates language and consciousness.")

# Access component-specific results
consciousness_level = result.get('consciousness_level')
neural_score = result.get('neural_linguistic_score')
recursive_depth = result.get('recursive_pattern_depth')
memory_associations = result.get('memory_associations')

# Adjust LLM weight
central_node.set_llm_weight(0.8)
```

## Database Synchronization System

The Enhanced Language System features a robust database synchronization mechanism implemented through the Language Database Bridge. This component ensures data consistency between the language database and the central database system.

### Language Database Bridge

The Language Database Bridge (`src/language/language_database_bridge.py`) provides bidirectional synchronization capabilities for maintaining data consistency across database components:

```python
from language.language_database_bridge import get_language_database_bridge

# Get the singleton instance of the bridge
bridge = get_language_database_bridge()

# Connect to the central database
from language.database_connection_manager import get_database_connection_manager
connection_manager = get_database_connection_manager()
connection_manager.register_component("language_database_bridge", bridge)

# Trigger immediate synchronization
bridge.sync_now()

# Get bridge status information
status = bridge.get_status()
print(f"Sync stats: {status['sync_stats']}")
```

### Key Features

1. **Singleton Pattern Implementation**
   - Ensures a single consistent instance throughout the application
   - Thread-safe implementation for reliable concurrent access
   - Centralized management of database connections

2. **Bidirectional Synchronization**
   - Synchronizes data from language database to central database
   - Supports synchronization from central database to language database
   - Configurable priorities for different data types

3. **Thread-Based Synchronization**
   - Background thread for periodic automatic synchronization
   - Customizable synchronization interval
   - On-demand synchronization capability

4. **Comprehensive Data Synchronization**
   - Conversation data synchronization with format conversion
   - Pattern detection sharing across database components
   - Learning statistics aggregation and integration
   - Historical data preservation and archiving

5. **Robust Error Handling**
   - Graceful recovery from connection failures
   - Detailed error logging and tracking
   - Automatic retry mechanisms for transient errors
   - Transaction-based updates for data integrity

6. **Monitoring and Metrics**
   - Detailed synchronization statistics and metrics collection
   - Status reporting for monitoring synchronization health
   - Performance metrics for optimization
   - Diagnostic capabilities for troubleshooting

### Database Connection Verification

The system includes a comprehensive database connection verification script (`src/verify_database_connections.py`) that ensures proper connectivity between all database components:

```bash
# Verify database connections
python src/verify_database_connections.py

# Force immediate synchronization
python src/verify_database_connections.py --sync-now

# Optimize database connections
python src/verify_database_connections.py --optimize

# Verify specific components
python src/verify_database_connections.py --components language_database_bridge central_language_node
```

This tool provides:
- Automatic discovery of database components
- Comprehensive connection verification
- Detailed status reporting
- Command-line options for advanced operations
- Integration with the system's logging infrastructure

### Recent Improvements

The database synchronization system has undergone significant enhancements:

1. **Enhanced Language Database Bridge**
   - Improved real-time synchronization capabilities
   - Enhanced conflict resolution with prioritization algorithms
   - Added support for partial and selective synchronization
   - Implementation of data transformation and normalization during sync

2. **Resilient Connection Management**
   - Automatic recovery from connection failures
   - Connection pooling for improved performance
   - Health monitoring with proactive connection maintenance
   - Intelligent retry mechanisms with exponential backoff

3. **Advanced Verification System**
   - Component auto-discovery with dynamic registration
   - Connection verification with detailed diagnostics
   - Synchronization status monitoring and reporting
   - Command-line interface with extended functionality

4. **Performance Optimizations**
   - Batch processing for improved synchronization speed
   - Optimized query patterns for reduced database load
   - Caching strategies for frequently accessed data
   - Resource management with connection lifetime control

The database synchronization system ensures reliable data persistence across all components of the Enhanced Language System, providing a foundation for consistent learning and memory capabilities.

## Mistral Integration

Our Enhanced Language System integrates with Mistral AI's powerful language models to provide more sophisticated responses while maintaining our system's unique capabilities.

### Features

- **Consciousness-Aware Responses**: Adjusts language model parameters based on consciousness measurements
- **Autowiki Learning System**: Self-improving knowledge base that grows with each interaction
- **Neural Metrics Integration**: Uses neural network metrics to guide language model responses
- **Streaming Response Support**: Progressive text generation with real-time metrics updates
- **Configurable Weights**: Adjustable weights for neural network vs. language model influence
- **Mock Mode Support**: Can run without API keys for development and testing

### Basic Usage

```python
from src.mistral_integration import MistralEnhancedSystem

# Initialize Mistral Enhanced System with your API key
system = MistralEnhancedSystem(
    api_key="your_mistral_api_key",
    model="mistral-small-latest",
    llm_weight=0.7,
    nn_weight=0.6
)

# Process messages
response = system.process_message("Tell me about consciousness in language systems")

# Get learning statistics
stats = system.get_system_stats()
print(f"Total exchanges: {stats['total_exchanges']}")
print(f"Average consciousness level: {stats['avg_consciousness_level']:.2f}")

# Close the system when done
system.close()
```

### Configuration Options

```python
from src.v7 import get_enhanced_language_integration

# Configure with specific options
config = {
    "api_key": "your_mistral_api_key",  # Optional if set as env var
    "model": "mistral-medium",          # Model choice
    "llm_weight": 0.7,                  # Weight for language model (0.0-1.0)
    "nn_weight": 0.6,                   # Weight for neural network (0.0-1.0)
    "learning_enabled": True,           # Enable autowiki learning
    "learning_dict_path": "data/my_custom_learning.json"  # Custom path
}

integration = get_enhanced_language_integration(
    mock_mode=False,  # False to use real API
    config=config
)
```

### Streaming Responses

```python
def handle_chunk(chunk, metrics):
    # Process each chunk of text as it arrives
    print(chunk, end="")
    # Metrics contains consciousness_level, neural_linguistic_score, etc.

# Process with streaming
full_response = integration.process_text_streaming(
    "Explain how neural networks and consciousness are related.",
    chunk_callback=handle_chunk
)
```

### Working with Autowiki

The Autowiki system allows the system to learn and improve over time:

```python
# Add knowledge to the system
integration.add_autowiki_entry(
    topic="Neural Networks",
    content="Neural networks are computing systems inspired by biological neural networks.",
    source="https://en.wikipedia.org/wiki/Neural_network"
)

# Retrieve knowledge
neural_info = integration.retrieve_autowiki("Neural Networks")
print(neural_info["content"])

# Search for knowledge
results = integration.search_autowiki("consciousness neural")
for result in results:
    print(f"{result['topic']}: {result['relevance']}")
```

### Available Models

- `mistral-tiny-latest` - Fastest, most efficient model
- `mistral-small-latest` - Good balance of speed and capability
- `mistral-medium-latest` - More powerful model, good for complex tasks
- `mistral-large-latest` - Most powerful model, slower processing

### Environment Setup

Set your Mistral API key as an environment variable:

```bash
# Linux/Mac
export MISTRAL_API_KEY=your_key_here

# Windows
set MISTRAL_API_KEY=your_key_here
```

## Command Line Interface

Run the chat interface with adjustable weights:

```bash
python src/chat_with_system.py --weight 0.7 --nnweight 0.6
```

Run with Mistral integration:

```bash
python src/mistral_integration.py --api-key YOUR_API_KEY --model mistral-medium-latest --weight 0.8 --nnweight 0.7
```

Try the demo with various options:

```bash
# Run basic demo
python src/v7/run_enhanced_mistral_demo.py

# Run in interactive mode
python src/v7/run_enhanced_mistral_demo.py --interactive

# Run with learning enabled
python src/v7/run_enhanced_mistral_demo.py --learning

# Run with custom model
python src/v7/run_enhanced_mistral_demo.py --model mistral-large-latest

# Run in mock mode (no API key needed)
python src/v7/run_enhanced_mistral_demo.py --mock
```

## PySide6 Graphical Interface

The system includes multiple PySide6-based graphical user interfaces for easier interaction:

```bash
# Run the Mistral Chat with full Language System integration
python run_mistral_language_app.py

# Run the standalone Enhanced Language System UI
python run_language_ui.py

# Run the PySide6 interface with onsite memory integration
python run_memory_app.py
```

The Enhanced Language System UI provides:
- Direct access to all language system components (Language Memory, Conscious Mirror Language, etc.)
- Visual interaction with each component's capabilities
- LLM and neural network weight controls
- Real-time status monitoring

The Mistral-Language integration provides:
- Enhanced message processing using the language system
- Context enhancement from language memory
- Consciousness-aware responses
- Full access to Mistral AI models

The onsite memory system provides persistent storage for conversations and knowledge, enabling the application to remember past interactions and use them for context in future conversations.

Alternatively, you can run the version without onsite memory:

```bash
# Run the PySide6 interface without the memory system
python run_mistral_app.py
```

Or use the standalone version:

```bash
# Run the standalone PySide6 interface (no dependencies on other system components)
python mistral_pyside_app_standalone.py
```

## Testing

Run the comprehensive test script to verify functionality:

```bash
python src/test_enhanced_language_system.py
```

Test the Mistral integration specifically:

```bash
python test_mistral_fixed.py
```

This will run tests for each component individually and test the integrated functionality.

## LLM Weight Adjustment

The system allows for adjusting the influence of Language Model suggestions through the LLM weight parameter:

- `0.0` - No LLM influence, only rule-based processing
- `0.5` - Balanced LLM and rule-based processing (default)
- `1.0` - Maximum LLM influence

Adjust this parameter to control how much the system relies on language model suggestions vs. rule-based processing.

## Known Issues and Limitations

- **Neural Network Integration**: The system currently shows an error processing "complexity" which affects the neural integration functionality. This may result in simulated rather than genuine neural processing.
- **Google API Key**: The system attempts to use a Google API key which appears to be invalid. For now, the system operates in simulation mode.
- **System Learning**: While the database and conversation memory components are storing information (as evidenced by concept IDs being added), the actual learning from this data needs improvement.
- **Response Variety**: The system currently produces similar responses to different inputs in some cases, indicating limitations in its contextual understanding.

## Performance Considerations

- The system creates background processes for maintaining component integration
- Memory usage increases with the size of text corpus processed
- Consider using smaller data samples for initial testing
- Database operations may slow down with very large conversation histories

## Architecture

The Mistral integration consists of three main components:

1. **EnhancedLanguageIntegration**: Handles consciousness metrics and neural processing
2. **MistralIntegration**: Manages API calls to Mistral AI and autowiki learning
3. **EnhancedLanguageMistralIntegration**: Combines both systems with weighted processing

Consciousness metrics inform the Mistral AI system about how to respond, adjusting parameters like temperature, max tokens, and system prompts based on the measured consciousness level.

## License

MIT

## Documentation

For more detailed documentation, see the [docs](docs/ENHANCED_LANGUAGE_SYSTEM.md) directory.
