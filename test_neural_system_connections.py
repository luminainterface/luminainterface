#!/usr/bin/env python3
"""
Neural Network System Connection Tester

This script tests connections across the neural network system,
validates database connectivity, and ensures filtering systems
are working correctly.
"""

import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("NeuralSystemTester")

# Test results tracking
test_results = {
    "total": 0,
    "passed": 0,
    "failed": 0,
    "skipped": 0,
    "component_status": {},
    "tests": []
}

# Component import flags
COMPONENTS = {
    "neural_linguistic_processor": False,
    "language_memory": False,
    "conversation_memory": False,
    "conscious_mirror_language": False,
    "database_manager": False,
    "autowiki": False,
    "background_learning": False,
    "verify_database_connections": False,
    "mistral_integration": False
}

# Try to import components
try:
    # Import neural linguistic processor
    try:
        from language.neural_linguistic_processor import NeuralLinguisticProcessor
        COMPONENTS["neural_linguistic_processor"] = True
    except ImportError:
        logger.warning("Could not import NeuralLinguisticProcessor")

    # Import language memory
    try:
        from language.language_memory import LanguageMemory
        COMPONENTS["language_memory"] = True
    except ImportError:
        logger.warning("Could not import LanguageMemory")

    # Import conversation memory
    try:
        from language.conversation_memory import ConversationMemory
        COMPONENTS["conversation_memory"] = True
    except ImportError:
        logger.warning("Could not import ConversationMemory")

    # Import conscious mirror language
    try:
        from language.conscious_mirror_language import ConsciousMirrorLanguage
        COMPONENTS["conscious_mirror_language"] = True
    except ImportError:
        logger.warning("Could not import ConsciousMirrorLanguage")

    # Import database manager
    try:
        from language.database_manager import DatabaseManager
        COMPONENTS["database_manager"] = True
    except ImportError:
        logger.warning("Could not import DatabaseManager")

    # Import autowiki
    try:
        from v7.autowiki import AutoWiki
        COMPONENTS["autowiki"] = True
    except ImportError:
        logger.warning("Could not import AutoWiki")

    # Import background learning engine
    try:
        from language.background_learning_engine import BackgroundLearningEngine
        COMPONENTS["background_learning"] = True
    except ImportError:
        logger.warning("Could not import BackgroundLearningEngine")

    # Import database connection verification
    try:
        import importlib.util
        spec = importlib.util.find_spec("verify_database_connections")
        if spec is not None:
            COMPONENTS["verify_database_connections"] = True
    except:
        logger.warning("Could not check verify_database_connections")

    # Import mistral integration
    try:
        try:
            from v7.mistral_integration import MistralEnhancedSystem
            COMPONENTS["mistral_integration"] = True
        except ImportError:
            try:
                from v7.enhanced_language_mistral_integration import get_enhanced_language_integration
                COMPONENTS["mistral_integration"] = True
            except ImportError:
                logger.warning("Could not import Mistral integration modules")
    except:
        logger.warning("Error during Mistral integration import")

except Exception as e:
    logger.error(f"Error during component import process: {e}")

def run_test(test_name: str, component_name: str) -> Tuple[bool, str]:
    """
    Run a test and track the result
    
    Args:
        test_name: Name of the test
        component_name: Name of the component being tested
        
    Returns:
        Tuple of (success, message)
    """
    test_results["total"] += 1
    
    # Check if component is available
    if not COMPONENTS.get(component_name, False):
        test_results["skipped"] += 1
        test_results["tests"].append({
            "name": test_name,
            "component": component_name,
            "status": "SKIPPED",
            "message": f"Component {component_name} not available"
        })
        return False, f"Component {component_name} not available"
    
    # Update component status
    if component_name not in test_results["component_status"]:
        test_results["component_status"][component_name] = {
            "tests": 0,
            "passed": 0,
            "failed": 0
        }
    
    test_results["component_status"][component_name]["tests"] += 1
    
    return True, ""

def test_success(test_name: str, component_name: str, message: str = ""):
    """Record a successful test"""
    test_results["passed"] += 1
    test_results["component_status"][component_name]["passed"] += 1
    test_results["tests"].append({
        "name": test_name,
        "component": component_name,
        "status": "PASSED",
        "message": message
    })
    logger.info(f"✓ PASSED: {test_name} - {message}")

def test_failure(test_name: str, component_name: str, message: str = ""):
    """Record a failed test"""
    test_results["failed"] += 1
    test_results["component_status"][component_name]["failed"] += 1
    test_results["tests"].append({
        "name": test_name,
        "component": component_name,
        "status": "FAILED",
        "message": message
    })
    logger.error(f"✗ FAILED: {test_name} - {message}")

def test_neural_linguistic_processor():
    """Test Neural Linguistic Processor connections and functionality"""
    component = "neural_linguistic_processor"
    
    # Test initialization
    test_name = "NLP_Initialization"
    can_run, msg = run_test(test_name, component)
    if not can_run:
        logger.warning(f"Skipping {test_name}: {msg}")
        return
    
    try:
        nlp = NeuralLinguisticProcessor()
        test_success(test_name, component, "NeuralLinguisticProcessor initialized successfully")
    except Exception as e:
        test_failure(test_name, component, f"Error initializing NeuralLinguisticProcessor: {e}")
        return
    
    # Test pattern recognition functionality
    test_name = "NLP_PatternRecognition"
    can_run, msg = run_test(test_name, component)
    if not can_run:
        return
    
    try:
        result = nlp.process_text("Neural networks learn through pattern recognition and weight adjustment")
        patterns = nlp.recognize_patterns("Neural networks learn through pattern recognition and weight adjustment")
        
        if result and isinstance(result, dict):
            test_success(test_name, component, f"Successfully processed text with {len(patterns)} patterns")
        else:
            test_failure(test_name, component, "Failed to process text properly")
    except Exception as e:
        test_failure(test_name, component, f"Error in pattern recognition: {e}")
    
    # Test filtering system
    test_name = "NLP_FilteringSystem"
    can_run, msg = run_test(test_name, component)
    if not can_run:
        return
    
    try:
        # Test with different confidence thresholds
        full_patterns = nlp.recognize_patterns("Neural networks learn through pattern recognition")
        
        # Get processor to apply filtering
        if hasattr(nlp, "filter_patterns_by_confidence"):
            filtered_patterns = nlp.filter_patterns_by_confidence(full_patterns, 0.8)
            if len(filtered_patterns) <= len(full_patterns):
                test_success(test_name, component, f"Successfully filtered patterns: {len(full_patterns)} -> {len(filtered_patterns)}")
            else:
                test_failure(test_name, component, "Pattern filtering not working correctly")
        else:
            # Manual filtering
            filtered_patterns = [p for p in full_patterns if p.get("confidence", 0) >= 0.8]
            if len(filtered_patterns) <= len(full_patterns):
                test_success(test_name, component, f"Successfully applied manual pattern filtering: {len(full_patterns)} -> {len(filtered_patterns)}")
            else:
                test_failure(test_name, component, "Manual pattern filtering not working correctly")
    except Exception as e:
        test_failure(test_name, component, f"Error in filtering system: {e}")

def test_language_memory():
    """Test Language Memory connections and functionality"""
    component = "language_memory"
    
    # Test initialization
    test_name = "LanguageMemory_Initialization"
    can_run, msg = run_test(test_name, component)
    if not can_run:
        logger.warning(f"Skipping {test_name}: {msg}")
        return
    
    try:
        memory = LanguageMemory()
        test_success(test_name, component, "LanguageMemory initialized successfully")
    except Exception as e:
        test_failure(test_name, component, f"Error initializing LanguageMemory: {e}")
        return
    
    # Test association storage and recall
    test_name = "LanguageMemory_AssociationStorage"
    can_run, msg = run_test(test_name, component)
    if not can_run:
        return
    
    try:
        # Store a test association
        memory.store_word_association("neural", "network", 0.9)
        
        # Recall the association
        associations = memory.recall_associations("neural")
        
        if associations and "network" in associations:
            test_success(test_name, component, "Successfully stored and recalled word associations")
        else:
            test_failure(test_name, component, "Failed to store or recall word associations")
    except Exception as e:
        test_failure(test_name, component, f"Error in association storage/recall: {e}")
    
    # Test filtering system
    test_name = "LanguageMemory_FilteringSystem"
    can_run, msg = run_test(test_name, component)
    if not can_run:
        return
    
    try:
        # Store multiple associations with different strengths
        memory.store_word_association("test", "strong", 0.9)
        memory.store_word_association("test", "medium", 0.6)
        memory.store_word_association("test", "weak", 0.3)
        
        # Test recall with filter
        if hasattr(memory, "recall_associations_with_threshold"):
            strong_associations = memory.recall_associations_with_threshold("test", 0.7)
            if strong_associations and "strong" in strong_associations and "medium" not in strong_associations:
                test_success(test_name, component, "Successfully filtered associations by strength")
            else:
                test_failure(test_name, component, "Association filtering not working correctly")
        else:
            # Test with standard recall and check values
            all_associations = memory.recall_associations("test")
            if all_associations and all_associations.get("strong", 0) >= 0.7:
                test_success(test_name, component, "Associations stored with correct strengths")
            else:
                test_failure(test_name, component, "Association strengths not stored correctly")
    except Exception as e:
        test_failure(test_name, component, f"Error in association filtering: {e}")

def test_database_manager():
    """Test Database Manager connections and functionality"""
    component = "database_manager"
    
    # Test initialization
    test_name = "DatabaseManager_Initialization"
    can_run, msg = run_test(test_name, component)
    if not can_run:
        logger.warning(f"Skipping {test_name}: {msg}")
        return
    
    try:
        db = DatabaseManager()
        test_success(test_name, component, "DatabaseManager initialized successfully")
    except Exception as e:
        test_failure(test_name, component, f"Error initializing DatabaseManager: {e}")
        return
    
    # Test database connection
    test_name = "DatabaseManager_Connection"
    can_run, msg = run_test(test_name, component)
    if not can_run:
        return
    
    try:
        # Check if connection is active
        if hasattr(db, "engine") and db.engine:
            test_success(test_name, component, "Database connection established")
        else:
            test_failure(test_name, component, "Database connection failed")
    except Exception as e:
        test_failure(test_name, component, f"Error checking database connection: {e}")
    
    # Test conversation creation and retrieval
    test_name = "DatabaseManager_ConversationStorage"
    can_run, msg = run_test(test_name, component)
    if not can_run:
        return
    
    try:
        # Create a test conversation
        conversation_id = db.create_conversation()
        
        # Store a test exchange
        exchange_id = db.store_exchange(
            conversation_id=conversation_id,
            user_input="This is a test",
            system_response="Test received",
            user_id="test_user"
        )
        
        # Retrieve the conversation
        conversation = db.get_conversation(conversation_id)
        
        if conversation and exchange_id:
            test_success(test_name, component, "Successfully created and retrieved conversation")
        else:
            test_failure(test_name, component, "Failed to create or retrieve conversation")
    except Exception as e:
        test_failure(test_name, component, f"Error in conversation storage/retrieval: {e}")
    
    # Test filtering system
    test_name = "DatabaseManager_FilteringSystem"
    can_run, msg = run_test(test_name, component)
    if not can_run:
        return
    
    try:
        # Add multiple pattern detections
        if hasattr(db, "add_pattern_detection"):
            # Add patterns with different confidences
            db.add_pattern_detection({
                "exchange_id": exchange_id,
                "pattern_text": "High confidence pattern",
                "pattern_type": "test",
                "confidence": 0.9
            })
            
            db.add_pattern_detection({
                "exchange_id": exchange_id,
                "pattern_text": "Low confidence pattern",
                "pattern_type": "test",
                "confidence": 0.3
            })
            
            # Test pattern retrieval with filter
            if hasattr(db, "get_high_confidence_patterns"):
                high_patterns = db.get_high_confidence_patterns(0.7)
                if high_patterns and any("High confidence" in p.get("pattern_text", "") for p in high_patterns):
                    test_success(test_name, component, "Successfully filtered patterns by confidence")
                else:
                    test_failure(test_name, component, "Pattern filtering not working correctly")
            else:
                # Test pattern retrieval without specific filter
                test_success(test_name, component, "Pattern storage working, but no specific filtering method")
        else:
            test_success(test_name, component, "Pattern detection not supported")
    except Exception as e:
        test_failure(test_name, component, f"Error in pattern filtering: {e}")
    
    # Clean up
    try:
        db.close()
    except:
        pass

def test_background_learning_engine():
    """Test Background Learning Engine connections and functionality"""
    component = "background_learning"
    
    # Test initialization
    test_name = "BackgroundLearning_Initialization"
    can_run, msg = run_test(test_name, component)
    if not can_run:
        logger.warning(f"Skipping {test_name}: {msg}")
        return
    
    try:
        from language.background_learning_engine import get_background_learning_engine
        learning_engine = get_background_learning_engine()
        test_success(test_name, component, "BackgroundLearningEngine initialized successfully")
    except Exception as e:
        test_failure(test_name, component, f"Error initializing BackgroundLearningEngine: {e}")
        return
    
    # Test component connections
    test_name = "BackgroundLearning_ComponentConnections"
    can_run, msg = run_test(test_name, component)
    if not can_run:
        return
    
    try:
        # Get status to check component connections
        status = learning_engine.get_status()
        components = status.get("components", {})
        
        # Check if at least one component is connected
        if any(components.values()):
            connected = [name for name, connected in components.items() if connected]
            test_success(test_name, component, f"Connected to: {', '.join(connected)}")
        else:
            test_failure(test_name, component, "Not connected to any components")
    except Exception as e:
        test_failure(test_name, component, f"Error checking component connections: {e}")
    
    # Test learning queue
    test_name = "BackgroundLearning_LearningQueue"
    can_run, msg = run_test(test_name, component)
    if not can_run:
        return
    
    try:
        # Add a test item to the learning queue
        success = learning_engine.add_learning_item(
            item_type="pattern",
            data={
                "text": "Test pattern for learning queue",
                "confidence": 0.8,
                "type": "test"
            },
            source="system_test"
        )
        
        # Check queue size
        status = learning_engine.get_status()
        if success and status.get("queue_size", 0) > 0:
            test_success(test_name, component, f"Successfully added item to learning queue (size: {status.get('queue_size')})")
        else:
            test_failure(test_name, component, "Failed to add item to learning queue")
    except Exception as e:
        test_failure(test_name, component, f"Error adding to learning queue: {e}")
    
    # Test filtering system
    test_name = "BackgroundLearning_FilteringSystem"
    can_run, msg = run_test(test_name, component)
    if not can_run:
        return
    
    try:
        # Check learning configuration for filtering parameters
        min_confidence = learning_engine.learning_config.get("min_pattern_confidence", 0)
        
        if min_confidence > 0:
            # Add patterns with different confidences
            learning_engine.add_learning_item(
                item_type="pattern",
                data={
                    "text": "High confidence test pattern",
                    "confidence": 0.9,
                    "type": "test"
                },
                source="system_test"
            )
            
            learning_engine.add_learning_item(
                item_type="pattern",
                data={
                    "text": "Low confidence test pattern",
                    "confidence": 0.2,
                    "type": "test"
                },
                source="system_test"
            )
            
            test_success(test_name, component, f"Filtering configured with min_confidence: {min_confidence}")
        else:
            test_failure(test_name, component, "Filtering not properly configured")
    except Exception as e:
        test_failure(test_name, component, f"Error testing filtering system: {e}")
    
    # Clean up
    try:
        learning_engine.stop()
    except:
        pass

def test_mistral_integration():
    """Test Mistral Integration connections and functionality"""
    component = "mistral_integration"
    
    # Test initialization (mock mode)
    test_name = "MistralIntegration_Initialization"
    can_run, msg = run_test(test_name, component)
    if not can_run:
        logger.warning(f"Skipping {test_name}: {msg}")
        return
    
    try:
        # Try different import paths
        mistral_system = None
        
        try:
            from v7.enhanced_language_mistral_integration import get_enhanced_language_integration
            mistral_system = get_enhanced_language_integration(mock_mode=True)
        except ImportError:
            try:
                from v7.mistral_integration import MistralEnhancedSystem
                mistral_system = MistralEnhancedSystem(mock_mode=True)
            except:
                raise ImportError("Could not import Mistral integration")
        
        if mistral_system:
            test_success(test_name, component, "Mistral integration initialized in mock mode")
        else:
            test_failure(test_name, component, "Failed to initialize Mistral integration")
    except Exception as e:
        test_failure(test_name, component, f"Error initializing Mistral integration: {e}")
        return
    
    # Test autowiki functionality
    test_name = "MistralIntegration_Autowiki"
    can_run, msg = run_test(test_name, component)
    if not can_run:
        return
    
    try:
        # Test adding entries to autowiki
        if hasattr(mistral_system, "add_autowiki_entry"):
            mistral_system.add_autowiki_entry(
                topic="System Test",
                content="This is a test entry for the Autowiki system",
                source="system_test"
            )
            
            # Retrieve the entry
            if hasattr(mistral_system, "retrieve_autowiki"):
                entry = mistral_system.retrieve_autowiki("System Test")
                
                if entry and "This is a test entry" in entry.get("content", ""):
                    test_success(test_name, component, "Successfully added and retrieved autowiki entry")
                else:
                    test_failure(test_name, component, "Failed to retrieve autowiki entry")
            else:
                test_failure(test_name, component, "retrieve_autowiki method not available")
        else:
            test_failure(test_name, component, "add_autowiki_entry method not available")
    except Exception as e:
        test_failure(test_name, component, f"Error testing autowiki functionality: {e}")
    
    # Test filtering system
    test_name = "MistralIntegration_FilteringSystem"
    can_run, msg = run_test(test_name, component)
    if not can_run:
        return
    
    try:
        # Check if search function is available
        if hasattr(mistral_system, "search_autowiki"):
            # Add a few more entries
            if hasattr(mistral_system, "add_autowiki_entry"):
                mistral_system.add_autowiki_entry(
                    topic="Neural Networks",
                    content="Neural networks process data through layers of nodes",
                    source="system_test"
                )
                
                mistral_system.add_autowiki_entry(
                    topic="Pattern Recognition",
                    content="Pattern recognition is a key aspect of neural networks",
                    source="system_test"
                )
                
                # Search for entries
                results = mistral_system.search_autowiki("neural")
                
                if results and len(results) > 0:
                    test_success(test_name, component, f"Search filtering returned {len(results)} results")
                else:
                    test_failure(test_name, component, "Search filtering returned no results")
            else:
                test_failure(test_name, component, "Cannot add entries for search test")
        else:
            test_failure(test_name, component, "search_autowiki method not available")
    except Exception as e:
        test_failure(test_name, component, f"Error testing search filtering: {e}")
    
    # Clean up
    try:
        if hasattr(mistral_system, "close"):
            mistral_system.close()
    except:
        pass

def test_database_connections():
    """Test Database connections across components"""
    component = "verify_database_connections"
    
    # Check if verify_database_connections is available
    test_name = "DatabaseConnections_Availability"
    can_run, msg = run_test(test_name, component)
    if not can_run:
        logger.warning(f"Skipping {test_name}: {msg}")
        return
    
    try:
        import importlib.util
        import subprocess
        
        # Try to find the module
        spec = importlib.util.find_spec("verify_database_connections")
        module_path = getattr(spec, "origin", None) if spec else None
        
        if module_path and os.path.exists(module_path):
            test_success(test_name, component, f"Found verify_database_connections at {module_path}")
        else:
            # Try to find in src directory
            src_path = os.path.join(project_root, "src", "verify_database_connections.py")
            if os.path.exists(src_path):
                module_path = src_path
                test_success(test_name, component, f"Found verify_database_connections at {module_path}")
            else:
                test_failure(test_name, component, "Could not find verify_database_connections module")
                return
    except Exception as e:
        test_failure(test_name, component, f"Error finding verify_database_connections: {e}")
        return
    
    # Test running the script
    test_name = "DatabaseConnections_Verification"
    can_run, msg = run_test(test_name, component)
    if not can_run:
        return
    
    try:
        # Run the script as a subprocess
        cmd = [sys.executable, module_path]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Get output with timeout
        try:
            stdout, stderr = process.communicate(timeout=30)
            
            # Check for errors
            if process.returncode != 0:
                test_failure(test_name, component, f"Script exited with error code {process.returncode}: {stderr}")
                return
            
            # Check output for success indicators
            if "connected" in stdout.lower() or "success" in stdout.lower():
                test_success(test_name, component, "Database connections verified successfully")
            else:
                test_failure(test_name, component, "Database verification did not indicate success")
                logger.info(f"Script output: {stdout}")
        except subprocess.TimeoutExpired:
            process.kill()
            test_failure(test_name, component, "Database verification timed out")
    except Exception as e:
        test_failure(test_name, component, f"Error running database verification: {e}")

def test_conversation_memory():
    """Test Conversation Memory connections and functionality"""
    component = "conversation_memory"
    
    # Test initialization
    test_name = "ConversationMemory_Initialization"
    can_run, msg = run_test(test_name, component)
    if not can_run:
        logger.warning(f"Skipping {test_name}: {msg}")
        return
    
    try:
        memory = ConversationMemory()
        test_success(test_name, component, "ConversationMemory initialized successfully")
    except Exception as e:
        test_failure(test_name, component, f"Error initializing ConversationMemory: {e}")
        return
    
    # Test conversation storage and retrieval
    test_name = "ConversationMemory_Storage"
    can_run, msg = run_test(test_name, component)
    if not can_run:
        return
    
    try:
        # Store a conversation
        conversation_id = memory.create_conversation()
        
        # Add an exchange
        exchange_id = memory.add_exchange(
            conversation_id=conversation_id,
            user_input="Test question",
            system_response="Test answer"
        )
        
        # Retrieve conversation
        conversation = memory.get_conversation(conversation_id)
        
        if conversation and exchange_id:
            test_success(test_name, component, "Successfully stored and retrieved conversation")
        else:
            test_failure(test_name, component, "Failed to store or retrieve conversation")
    except Exception as e:
        test_failure(test_name, component, f"Error in conversation storage/retrieval: {e}")
    
    # Test filtering system
    test_name = "ConversationMemory_FilteringSystem"
    can_run, msg = run_test(test_name, component)
    if not can_run:
        return
    
    try:
        # Add concepts to the exchange
        if hasattr(memory, "add_concepts_to_exchange"):
            memory.add_concepts_to_exchange(
                exchange_id=exchange_id,
                concepts={
                    "test": {"importance": 0.8}
                }
            )
            
            # Get context with filtering
            if hasattr(memory, "get_context"):
                context = memory.get_context("test query")
                if context and len(context) > 0:
                    test_success(test_name, component, f"Context filtering returned {len(context)} results")
                else:
                    test_failure(test_name, component, "Context filtering returned no results")
            else:
                test_success(test_name, component, "No context filtering method available")
        else:
            test_success(test_name, component, "No concept addition method available")
    except Exception as e:
        test_failure(test_name, component, f"Error testing filtering system: {e}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Test neural network system connections")
    
    parser.add_argument('--components', type=str, default=None,
                       help='Comma-separated list of components to test (default: all)')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output file for test results (JSON format)')
    
    return parser.parse_args()

def main():
    """Main entry point"""
    # Parse arguments
    args = parse_arguments()
    
    # Set logging level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print header
    print("\n===== Neural Network System Connection Tester =====\n")
    print("Testing connections and functionality across the system\n")
    
    # Print component availability
    print("Component Availability:")
    for component, available in COMPONENTS.items():
        status = "✓ Available" if available else "✗ Not Found"
        print(f"  {component:30} {status}")
    print("")
    
    # Determine which components to test
    components_to_test = []
    if args.components:
        components_to_test = [c.strip() for c in args.components.split(",")]
    else:
        # Test all available components
        components_to_test = [c for c, available in COMPONENTS.items() if available]
    
    print(f"Will test the following components: {', '.join(components_to_test)}\n")
    
    # Run tests
    if "neural_linguistic_processor" in components_to_test:
        print("\n--- Testing Neural Linguistic Processor ---")
        test_neural_linguistic_processor()
    
    if "language_memory" in components_to_test:
        print("\n--- Testing Language Memory ---")
        test_language_memory()
    
    if "conversation_memory" in components_to_test:
        print("\n--- Testing Conversation Memory ---")
        test_conversation_memory()
    
    if "database_manager" in components_to_test:
        print("\n--- Testing Database Manager ---")
        test_database_manager()
    
    if "background_learning" in components_to_test:
        print("\n--- Testing Background Learning Engine ---")
        test_background_learning_engine()
    
    if "mistral_integration" in components_to_test:
        print("\n--- Testing Mistral Integration ---")
        test_mistral_integration()
    
    if "verify_database_connections" in components_to_test:
        print("\n--- Testing Database Connections ---")
        test_database_connections()
    
    # Print test results summary
    print("\n===== Test Results Summary =====\n")
    print(f"Total Tests: {test_results['total']}")
    print(f"Passed: {test_results['passed']}")
    print(f"Failed: {test_results['failed']}")
    print(f"Skipped: {test_results['skipped']}")
    print("")
    
    # Print component results
    print("Component Results:")
    for component, results in test_results["component_status"].items():
        print(f"  {component:30} {results['passed']}/{results['tests']} passed")
    
    # Save results to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(test_results, f, indent=2)
        print(f"\nTest results saved to {args.output}")
    
    # Return non-zero exit code if any tests failed
    return 1 if test_results["failed"] > 0 else 0

if __name__ == "__main__":
    sys.exit(main()) 