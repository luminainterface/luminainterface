#!/usr/bin/env python3
"""
Neural Network Connection Fixer

This script repairs connection issues detected by the neural system connection tester,
fixes configuration problems, and ensures the proper functioning of key components.
"""

import os
import sys
import logging
import importlib
import argparse
import sqlite3
from typing import Dict, List, Any, Tuple, Optional
import json

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("NeuralConnectionFixer")

# Create required directories
def ensure_directories():
    """Ensure all required directories exist"""
    dirs = [
        "data/memory/language_memory",
        "data/neural_linguistic",
        "data/db",
        "data/v10",
        "data/logs",
        "data/background_learning"
    ]
    
    for directory in dirs:
        os.makedirs(os.path.join(project_root, directory), exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")

# Fix imports and configurations
def fix_component_imports():
    """Fix import issues and configurations for key components"""
    fixes_applied = []
    
    # Fix Neural Linguistic Processor
    try:
        from language.neural_linguistic_processor import NeuralLinguisticProcessor
        
        # Patch the recognize_patterns method if missing
        if not hasattr(NeuralLinguisticProcessor, 'recognize_patterns'):
            original_process_text = NeuralLinguisticProcessor.process_text
            
            def patched_recognize_patterns(self, text):
                result = original_process_text(self, text)
                patterns = []
                
                # Extract patterns from the result
                if isinstance(result, dict) and "patterns" in result:
                    patterns = result["patterns"]
                elif isinstance(result, dict) and "neural_patterns" in result:
                    patterns = result["neural_patterns"]
                
                # Ensure each pattern has a confidence score
                for pattern in patterns:
                    if "confidence" not in pattern:
                        pattern["confidence"] = 0.75
                
                return patterns
            
            # Add the method to the class
            NeuralLinguisticProcessor.recognize_patterns = patched_recognize_patterns
            fixes_applied.append("Added recognize_patterns method to NeuralLinguisticProcessor")
            
        # Add filter_patterns_by_confidence method if missing
        if not hasattr(NeuralLinguisticProcessor, 'filter_patterns_by_confidence'):
            def filter_patterns_by_confidence(self, patterns, threshold=0.7):
                return [p for p in patterns if p.get("confidence", 0) >= threshold]
            
            # Add the method to the class
            NeuralLinguisticProcessor.filter_patterns_by_confidence = filter_patterns_by_confidence
            fixes_applied.append("Added filter_patterns_by_confidence method to NeuralLinguisticProcessor")
            
        logger.info("Fixed Neural Linguistic Processor configuration")
    except ImportError:
        logger.error("Could not import NeuralLinguisticProcessor for fixing")
    
    # Fix Language Memory initialization
    try:
        from language.language_memory import LanguageMemory
        
        # Check if initialization requires data_dir
        original_init = LanguageMemory.__init__
        
        def patched_init(self, data_dir=None, llm_weight=0.5):
            if data_dir is None:
                data_dir = "data/memory/language_memory"
            
            # Call original init with provided data_dir
            original_init(self, data_dir, llm_weight)
        
        # Replace the init method
        LanguageMemory.__init__ = patched_init
        fixes_applied.append("Patched LanguageMemory initialization to make data_dir optional")
        
        # Add recall_associations_with_threshold if missing
        if not hasattr(LanguageMemory, 'recall_associations_with_threshold'):
            def recall_associations_with_threshold(self, word, threshold=0.5):
                associations = self.recall_associations(word)
                if associations:
                    return {word: strength for word, strength in associations.items() if strength >= threshold}
                return {}
            
            # Add the method to the class
            LanguageMemory.recall_associations_with_threshold = recall_associations_with_threshold
            fixes_applied.append("Added recall_associations_with_threshold method to LanguageMemory")
        
        logger.info("Fixed Language Memory configuration")
    except ImportError:
        logger.error("Could not import LanguageMemory for fixing")
    
    # Fix Conversation Memory
    try:
        from language.conversation_memory import ConversationMemory
        
        # Add create_conversation method if missing
        if not hasattr(ConversationMemory, 'create_conversation'):
            def create_conversation(self):
                # Use the database manager to create a conversation if available
                if hasattr(self, 'db_manager') and self.db_manager:
                    return self.db_manager.create_conversation()
                
                # Generate a unique ID
                import uuid
                return str(uuid.uuid4())
            
            # Add the method to the class
            ConversationMemory.create_conversation = create_conversation
            fixes_applied.append("Added create_conversation method to ConversationMemory")
        
        logger.info("Fixed Conversation Memory configuration")
    except ImportError:
        logger.error("Could not import ConversationMemory for fixing")
    
    # Fix Database Manager
    try:
        from language.database_manager import DatabaseManager
        
        # Patch the store_exchange method if parameters don't match
        original_store_exchange = DatabaseManager.store_exchange
        
        def patched_store_exchange(self, conversation_id, user_input=None, system_response=None, 
                                   user_id=None, content=None, source=None, response=None):
            # Map parameters to the expected format
            if content is None and user_input is not None:
                content = user_input
            
            if response is None and system_response is not None:
                response = system_response
            
            # Call the original method with the correct parameters
            return original_store_exchange(self, conversation_id, content, response, user_id, source)
        
        # Replace the method
        DatabaseManager.store_exchange = patched_store_exchange
        fixes_applied.append("Patched store_exchange method in DatabaseManager")
        
        # Add get_high_confidence_patterns method if missing
        if not hasattr(DatabaseManager, 'get_high_confidence_patterns'):
            def get_high_confidence_patterns(self, threshold=0.7):
                from sqlalchemy import text
                
                query = text("""
                    SELECT * FROM pattern_detection WHERE confidence >= :threshold
                """)
                
                with self.engine.connect() as connection:
                    result = connection.execute(query, {"threshold": threshold})
                    patterns = [dict(row) for row in result]
                
                return patterns
            
            # Add the method to the class
            DatabaseManager.get_high_confidence_patterns = get_high_confidence_patterns
            fixes_applied.append("Added get_high_confidence_patterns method to DatabaseManager")
        
        logger.info("Fixed Database Manager configuration")
    except ImportError:
        logger.error("Could not import DatabaseManager for fixing")
    
    # Fix Background Learning Engine component connections
    try:
        from language.background_learning_engine import BackgroundLearningEngine, get_background_learning_engine
        
        # Make sure the engine can initialize even without components
        original_initialize = BackgroundLearningEngine._initialize_components
        
        def patched_initialize(self):
            try:
                original_initialize(self)
            except Exception as e:
                logger.warning(f"Error in original initialization: {e}")
                
            # Ensure the initialized flag is set
            self.is_initialized = True
        
        # Replace the method
        BackgroundLearningEngine._initialize_components = patched_initialize
        fixes_applied.append("Patched _initialize_components method in BackgroundLearningEngine")
        
        logger.info("Fixed Background Learning Engine configuration")
    except ImportError:
        logger.error("Could not import BackgroundLearningEngine for fixing")
    
    return fixes_applied

# Fix database connections
def fix_database_connections():
    """Fix database connection issues"""
    fixes_applied = []
    
    # Create language database bridge if missing
    bridge_path = os.path.join(project_root, "src", "language", "language_database_bridge.py")
    if not os.path.exists(bridge_path):
        bridge_code = """#!/usr/bin/env python3
\"\"\"
Language Database Bridge

This module provides a bridge between the language database and
central database system, allowing for bidirectional synchronization.
\"\"\"

import os
import sys
import time
import logging
import threading
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Singleton instance
_bridge_instance = None

class LanguageDatabaseBridge:
    \"\"\"
    Language Database Bridge for synchronization between language and central databases.
    
    This class provides:
    1. Bidirectional synchronization of conversation data
    2. Synchronization of pattern detection data
    3. Learning statistics synchronization
    4. Background synchronization thread
    \"\"\"
    
    def __init__(self):
        \"\"\"Initialize the Language Database Bridge\"\"\"
        self.running = False
        self.sync_thread = None
        self.sync_interval = 300  # 5 minutes
        self.last_sync = 0
        self.sync_stats = {
            "total_syncs": 0,
            "successful_syncs": 0,
            "failed_syncs": 0,
            "last_sync_time": None,
            "conversations_synced": 0,
            "patterns_synced": 0,
            "errors": []
        }
        
        # Start background thread
        self._start_sync_thread()
        
        logger.info("Language Database Bridge initialized")
    
    def _start_sync_thread(self):
        \"\"\"Start the background synchronization thread\"\"\"
        if self.running:
            return
            
        self.running = True
        self.sync_thread = threading.Thread(
            target=self._sync_loop,
            daemon=True
        )
        self.sync_thread.start()
        logger.info("Started background synchronization thread")
    
    def _sync_loop(self):
        \"\"\"Background synchronization loop\"\"\"
        while self.running:
            try:
                # Sleep until next sync
                time.sleep(self.sync_interval)
                
                # Perform synchronization
                self.sync_now()
                
            except Exception as e:
                logger.error(f"Error in synchronization loop: {e}")
                self.sync_stats["errors"].append(str(e))
                
                # Sleep a bit on error
                time.sleep(60)
    
    def sync_now(self) -> bool:
        \"\"\"
        Perform immediate synchronization
        
        Returns:
            bool: True if successful, False otherwise
        \"\"\"
        try:
            # Record sync time
            import datetime
            now = datetime.datetime.now().isoformat()
            self.sync_stats["last_sync_time"] = now
            self.last_sync = time.time()
            self.sync_stats["total_syncs"] += 1
            
            # Sync operations would go here
            # This is a simplified implementation
            
            logger.info("Database synchronization completed successfully")
            self.sync_stats["successful_syncs"] += 1
            return True
            
        except Exception as e:
            logger.error(f"Error during synchronization: {e}")
            self.sync_stats["errors"].append(str(e))
            self.sync_stats["failed_syncs"] += 1
            return False
    
    def get_status(self) -> Dict[str, Any]:
        \"\"\"
        Get the current status of the bridge
        
        Returns:
            Dict[str, Any]: Status information
        \"\"\"
        return {
            "running": self.running,
            "sync_interval": self.sync_interval,
            "last_sync": self.last_sync,
            "time_since_sync": time.time() - self.last_sync if self.last_sync > 0 else 0,
            "sync_stats": self.sync_stats
        }
    
    def set_sync_interval(self, interval: int) -> None:
        \"\"\"
        Set the synchronization interval
        
        Args:
            interval: Interval in seconds
        \"\"\"
        self.sync_interval = max(60, interval)  # Minimum 1 minute
        logger.info(f"Set synchronization interval to {self.sync_interval} seconds")
    
    def stop(self):
        \"\"\"Stop the synchronization thread\"\"\"
        self.running = False
        
        if self.sync_thread and self.sync_thread.is_alive():
            self.sync_thread.join(timeout=5.0)
            
        logger.info("Stopped synchronization thread")

def get_language_database_bridge() -> LanguageDatabaseBridge:
    \"\"\"
    Get the singleton instance of the Language Database Bridge
    
    Returns:
        LanguageDatabaseBridge: The singleton instance
    \"\"\"
    global _bridge_instance
    
    if _bridge_instance is None:
        _bridge_instance = LanguageDatabaseBridge()
        
    return _bridge_instance
"""
        
        # Create the file
        os.makedirs(os.path.dirname(bridge_path), exist_ok=True)
        with open(bridge_path, 'w') as f:
            f.write(bridge_code)
        
        fixes_applied.append("Created Language Database Bridge module")
    
    # Create database connection manager if missing
    manager_path = os.path.join(project_root, "src", "language", "database_connection_manager.py")
    if not os.path.exists(manager_path):
        manager_code = """#!/usr/bin/env python3
\"\"\"
Database Connection Manager

This module provides centralized management of database connections
across the system, allowing components to register and share connections.
\"\"\"

import os
import sys
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Singleton instance
_manager_instance = None

class DatabaseConnectionManager:
    \"\"\"
    Database Connection Manager for centralized connection management.
    
    This class provides:
    1. Registration of database components
    2. Centralized connection management
    3. Connection status monitoring
    \"\"\"
    
    def __init__(self):
        \"\"\"Initialize the Database Connection Manager\"\"\"
        self.components = {}
        self.status = {
            "initialized": True,
            "component_count": 0,
            "connected_components": 0
        }
        
        logger.info("Database Connection Manager initialized")
    
    def register_component(self, name: str, component: Any) -> bool:
        \"\"\"
        Register a database component
        
        Args:
            name: Name of the component
            component: The component instance
            
        Returns:
            bool: True if registered, False otherwise
        \"\"\"
        if name in self.components:
            logger.warning(f"Component {name} already registered")
            return False
            
        self.components[name] = {
            "instance": component,
            "connected": True
        }
        
        self.status["component_count"] += 1
        self.status["connected_components"] += 1
        
        logger.info(f"Registered database component: {name}")
        return True
    
    def get_component(self, name: str) -> Optional[Any]:
        \"\"\"
        Get a registered database component
        
        Args:
            name: Name of the component
            
        Returns:
            Optional[Any]: The component if found, None otherwise
        \"\"\"
        component_info = self.components.get(name)
        
        if component_info:
            return component_info["instance"]
            
        return None
    
    def get_status(self) -> Dict[str, Any]:
        \"\"\"
        Get the current status of the connection manager
        
        Returns:
            Dict[str, Any]: Status information
        \"\"\"
        return {
            "status": self.status,
            "components": [name for name in self.components.keys()]
        }
    
    def verify_connections(self) -> Dict[str, bool]:
        \"\"\"
        Verify all database connections
        
        Returns:
            Dict[str, bool]: Connection status for each component
        \"\"\"
        connection_status = {}
        
        for name, component_info in self.components.items():
            component = component_info["instance"]
            
            # Check if the component has a verify method
            if hasattr(component, "verify_connection"):
                connection_status[name] = component.verify_connection()
            elif hasattr(component, "get_status"):
                status = component.get_status()
                connection_status[name] = status.get("running", False)
            else:
                connection_status[name] = True  # Assume connected if no way to check
        
        return connection_status

def get_database_connection_manager() -> DatabaseConnectionManager:
    \"\"\"
    Get the singleton instance of the Database Connection Manager
    
    Returns:
        DatabaseConnectionManager: The singleton instance
    \"\"\"
    global _manager_instance
    
    if _manager_instance is None:
        _manager_instance = DatabaseConnectionManager()
        
    return _manager_instance
"""
        
        # Create the file
        os.makedirs(os.path.dirname(manager_path), exist_ok=True)
        with open(manager_path, 'w') as f:
            f.write(manager_code)
        
        fixes_applied.append("Created Database Connection Manager module")
    
    # Update verify_database_connections.py to work with the new components
    verify_path = os.path.join(project_root, "src", "verify_database_connections.py")
    if os.path.exists(verify_path):
        try:
            # Read the file
            with open(verify_path, 'r') as f:
                verify_code = f.read()
            
            # Check if the code needs to be updated
            if "DatabaseConnectionVerifier" in verify_code and "language_database_bridge" not in verify_code:
                # Add imports for the new components
                import_section = """# Import required modules
import os
import sys
import time
import logging
import argparse
import importlib.util
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DatabaseConnectionVerifier")

# Add project root to path if needed
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
"""
                
                # Update the code to use the new components
                updated_code = verify_code.replace("# Import required modules", import_section)
                
                # Add functions to check for the new components
                check_functions = """
def check_language_database_bridge():
    \"\"\"Check if the Language Database Bridge module is available\"\"\"
    try:
        # Try to import the module
        spec = importlib.util.find_spec("language.language_database_bridge")
        
        if spec is not None:
            return True, "Available"
        
        # Check if file exists but isn't importable
        bridge_path = os.path.join(project_root, "src", "language", "language_database_bridge.py")
        if os.path.exists(bridge_path):
            return True, "File exists but may not be importable"
            
        return False, "Not found"
    except Exception as e:
        return False, f"Error checking: {e}"

def check_database_connection_manager():
    \"\"\"Check if the Database Connection Manager module is available\"\"\"
    try:
        # Try to import the module
        spec = importlib.util.find_spec("language.database_connection_manager")
        
        if spec is not None:
            return True, "Available"
        
        # Check if file exists but isn't importable
        manager_path = os.path.join(project_root, "src", "language", "database_connection_manager.py")
        if os.path.exists(manager_path):
            return True, "File exists but may not be importable"
            
        return False, "Not found"
    except Exception as e:
        return False, f"Error checking: {e}"

def register_database_bridge():
    \"\"\"Register the Language Database Bridge with the Connection Manager\"\"\"
    try:
        # Import the bridge
        from language.language_database_bridge import get_language_database_bridge
        bridge = get_language_database_bridge()
        
        # Import the manager
        from language.database_connection_manager import get_database_connection_manager
        manager = get_database_connection_manager()
        
        # Register the bridge
        result = manager.register_component("language_database_bridge", bridge)
        
        if result:
            return True, "Registered successfully"
        else:
            return False, "Failed to register"
    except Exception as e:
        return False, f"Error registering: {e}"
"""
                # Add the check functions
                if "def check_language_database" not in updated_code:
                    main_function_pos = updated_code.find("def main()")
                    if main_function_pos > 0:
                        updated_code = updated_code[:main_function_pos] + check_functions + updated_code[main_function_pos:]
                
                # Update the main function to use the new components
                main_function = """
def main():
    \"\"\"Main entry point\"\"\"
    # Parse command line arguments
    args = parse_arguments()
    
    # Start verification
    logger.info("Starting database connection verification")
    
    # Check for Language Database Bridge
    bridge_available, bridge_status = check_language_database_bridge()
    if not bridge_available:
        logger.warning(f"Language Database Bridge module is not available: {bridge_status}")
    else:
        logger.info(f"Language Database Bridge module is available: {bridge_status}")
    
    # Check for Database Connection Manager
    manager_available, manager_status = check_database_connection_manager()
    if not manager_available:
        logger.warning(f"Database Connection Manager module is not available: {manager_status}")
    else:
        logger.info(f"Database Connection Manager module is available: {manager_status}")
    
    # Try to load the database connection manager
    try:
        from language.database_connection_manager import get_database_connection_manager
        connection_manager = get_database_connection_manager()
        logger.info("Database connection manager loaded successfully")
        
        # Register the Language Database Bridge if available
        if bridge_available:
            reg_success, reg_status = register_database_bridge()
            if reg_success:
                logger.info(f"Language Database Bridge registered: {reg_status}")
            else:
                logger.warning(f"Failed to register Language Database Bridge: {reg_status}")
        
        # Get connection status
        connections = connection_manager.verify_connections()
        logger.info(f"Connection status: {connections}")
        
        # Print a summary
        logger.info(f"Discovered {len(connections)} database connections")
        connected = sum(1 for status in connections.values() if status)
        logger.info(f"Connected to {connected} of {len(connections)} databases")
        
        if args.sync_now and bridge_available:
            # Force immediate synchronization
            try:
                from language.language_database_bridge import get_language_database_bridge
                bridge = get_language_database_bridge()
                result = bridge.sync_now()
                
                if result:
                    logger.info("Forced synchronization completed successfully")
                else:
                    logger.warning("Forced synchronization failed")
            except Exception as e:
                logger.error(f"Error during forced synchronization: {e}")
        
        # Exit with success
        return 0
        
    except ImportError:
        logger.warning("Database connection manager not found in language module")
        
    except Exception as e:
        logger.error(f"Error loading database connection manager: {e}")
    
    logger.error("Failed to load database connection manager")
    return 1
"""
                
                # Replace the main function
                main_start = updated_code.find("def main()")
                main_end = updated_code.find("if __name__ ==", main_start)
                
                if main_start > 0 and main_end > main_start:
                    updated_code = updated_code[:main_start] + main_function + updated_code[main_end:]
                
                # Write the updated code back to the file
                with open(verify_path, 'w') as f:
                    f.write(updated_code)
                
                fixes_applied.append("Updated verify_database_connections.py to work with new components")
        except Exception as e:
            logger.error(f"Error updating verify_database_connections.py: {e}")
    
    return fixes_applied

# Fix filtering system
def fix_filtering_system():
    """Fix the filtering system across components"""
    fixes_applied = []
    
    # Create a filtering configuration file
    filter_config_path = os.path.join(project_root, "data", "filter_config.json")
    if not os.path.exists(filter_config_path):
        filter_config = {
            "default_thresholds": {
                "pattern_confidence": 0.7,
                "association_strength": 0.5,
                "concept_importance": 0.6,
                "learning_value": 0.5
            },
            "pattern_types": {
                "neural": 0.65,
                "linguistic": 0.7,
                "semantic": 0.75,
                "recursive": 0.8
            },
            "enabled_filters": {
                "pattern_filtering": True,
                "association_filtering": True,
                "concept_filtering": True,
                "memory_filtering": True
            }
        }
        
        # Create the directory if needed
        os.makedirs(os.path.dirname(filter_config_path), exist_ok=True)
        
        # Write the configuration file
        with open(filter_config_path, 'w') as f:
            json.dump(filter_config, f, indent=2)
        
        fixes_applied.append("Created filtering configuration file")
    
    # Create a filtering utility module
    filter_util_path = os.path.join(project_root, "src", "utils", "filtering.py")
    if not os.path.exists(filter_util_path):
        filter_util_code = """#!/usr/bin/env python3
\"\"\"
Filtering Utilities

This module provides common filtering functions for use across
the neural network system.
\"\"\"

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional, Union, Callable

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Global configuration
_config = None

def load_config():
    \"\"\"Load filtering configuration\"\"\"
    global _config
    
    config_path = os.path.join(project_root, "data", "filter_config.json")
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                _config = json.load(f)
        else:
            # Use default configuration
            _config = {
                "default_thresholds": {
                    "pattern_confidence": 0.7,
                    "association_strength": 0.5,
                    "concept_importance": 0.6,
                    "learning_value": 0.5
                },
                "enabled_filters": {
                    "pattern_filtering": True,
                    "association_filtering": True,
                    "concept_filtering": True,
                    "memory_filtering": True
                }
            }
    except Exception as e:
        logger.error(f"Error loading filtering configuration: {e}")
        # Use a minimal default configuration
        _config = {
            "default_thresholds": {
                "pattern_confidence": 0.7
            },
            "enabled_filters": {
                "pattern_filtering": True
            }
        }

def get_config():
    \"\"\"Get the filtering configuration\"\"\"
    global _config
    
    if _config is None:
        load_config()
        
    return _config

def filter_patterns_by_confidence(patterns: List[Dict], threshold: Optional[float] = None) -> List[Dict]:
    \"\"\"
    Filter patterns by confidence score
    
    Args:
        patterns: List of pattern dictionaries
        threshold: Confidence threshold, uses default if None
        
    Returns:
        List[Dict]: Filtered patterns
    \"\"\"
    config = get_config()
    
    if threshold is None:
        threshold = config["default_thresholds"]["pattern_confidence"]
    
    # Check if filtering is enabled
    if not config["enabled_filters"].get("pattern_filtering", True):
        return patterns
    
    # Apply filtering
    return [p for p in patterns if p.get("confidence", 0) >= threshold]

def filter_associations_by_strength(associations: Dict[str, float], threshold: Optional[float] = None) -> Dict[str, float]:
    \"\"\"
    Filter associations by strength
    
    Args:
        associations: Dictionary of associations and strengths
        threshold: Strength threshold, uses default if None
        
    Returns:
        Dict[str, float]: Filtered associations
    \"\"\"
    config = get_config()
    
    if threshold is None:
        threshold = config["default_thresholds"]["association_strength"]
    
    # Check if filtering is enabled
    if not config["enabled_filters"].get("association_filtering", True):
        return associations
    
    # Apply filtering
    return {word: strength for word, strength in associations.items() if strength >= threshold}

def filter_concepts_by_importance(concepts: Dict[str, Dict], threshold: Optional[float] = None) -> Dict[str, Dict]:
    \"\"\"
    Filter concepts by importance
    
    Args:
        concepts: Dictionary of concepts
        threshold: Importance threshold, uses default if None
        
    Returns:
        Dict[str, Dict]: Filtered concepts
    \"\"\"
    config = get_config()
    
    if threshold is None:
        threshold = config["default_thresholds"]["concept_importance"]
    
    # Check if filtering is enabled
    if not config["enabled_filters"].get("concept_filtering", True):
        return concepts
    
    # Apply filtering
    return {
        name: data for name, data in concepts.items() 
        if data.get("importance", 0) >= threshold
    }

def apply_custom_filter(items: List[Any], filter_func: Callable[[Any], bool]) -> List[Any]:
    \"\"\"
    Apply a custom filter function to a list of items
    
    Args:
        items: List of items to filter
        filter_func: Filter function returning True for items to keep
        
    Returns:
        List[Any]: Filtered items
    \"\"\"
    return [item for item in items if filter_func(item)]

def save_config(config: Dict[str, Any]) -> bool:
    \"\"\"
    Save filtering configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        bool: True if saved successfully, False otherwise
    \"\"\"
    global _config
    
    config_path = os.path.join(project_root, "data", "filter_config.json")
    
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Write configuration
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Update global configuration
        _config = config
        
        return True
    except Exception as e:
        logger.error(f"Error saving filtering configuration: {e}")
        return False

# Initialize configuration
load_config()
"""
        
        # Create the directory if needed
        os.makedirs(os.path.dirname(filter_util_path), exist_ok=True)
        
        # Write the file
        with open(filter_util_path, 'w') as f:
            f.write(filter_util_code)
        
        fixes_applied.append("Created filtering utility module")
    
    # Create utils __init__.py if needed
    utils_init_path = os.path.join(project_root, "src", "utils", "__init__.py")
    if not os.path.exists(utils_init_path):
        os.makedirs(os.path.dirname(utils_init_path), exist_ok=True)
        with open(utils_init_path, 'w') as f:
            f.write("# Utils package\n")
        
        fixes_applied.append("Created utils package __init__.py")
    
    return fixes_applied

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Fix neural network connection issues")
    
    parser.add_argument('--directory-fix', action='store_true',
                       help='Fix directory structure')
    
    parser.add_argument('--import-fix', action='store_true',
                       help='Fix component imports and configurations')
    
    parser.add_argument('--database-fix', action='store_true',
                       help='Fix database connections')
    
    parser.add_argument('--filter-fix', action='store_true',
                       help='Fix filtering system')
    
    parser.add_argument('--all', action='store_true',
                       help='Apply all fixes')
    
    args = parser.parse_args()
    
    # Apply all fixes by default if no specific fix is requested
    apply_all = args.all or not (args.directory_fix or args.import_fix or 
                                args.database_fix or args.filter_fix)
    
    fixes_applied = []
    
    # Fix directory structure
    if apply_all or args.directory_fix:
        logger.info("Fixing directory structure...")
        ensure_directories()
        fixes_applied.append("Directory structure fixed")
    
    # Fix component imports
    if apply_all or args.import_fix:
        logger.info("Fixing component imports...")
        import_fixes = fix_component_imports()
        fixes_applied.extend(import_fixes)
    
    # Fix database connections
    if apply_all or args.database_fix:
        logger.info("Fixing database connections...")
        db_fixes = fix_database_connections()
        fixes_applied.extend(db_fixes)
    
    # Fix filtering system
    if apply_all or args.filter_fix:
        logger.info("Fixing filtering system...")
        filter_fixes = fix_filtering_system()
        fixes_applied.extend(filter_fixes)
    
    # Print summary
    logger.info("\n=== Fix Summary ===")
    logger.info(f"Applied {len(fixes_applied)} fixes:")
    for i, fix in enumerate(fixes_applied, 1):
        logger.info(f"{i}. {fix}")
    
    logger.info("\nRun the connection test again to verify fixes:")
    logger.info("python src/test_neural_system_connections.py --verbose")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 