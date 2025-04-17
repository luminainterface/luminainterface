import os
import json
import time
import shutil
import logging
import pickle
import threading
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class PersistenceConfig:
    """Configuration settings for the persistence layer"""
    
    def __init__(
        self,
        base_directory: str = "data",
        autosave_interval: int = 300,  # 5 minutes
        max_snapshots: int = 5,
        compression_enabled: bool = True,
        snapshot_on_shutdown: bool = True
    ):
        self.base_directory = base_directory
        self.autosave_interval = autosave_interval
        self.max_snapshots = max_snapshots
        self.compression_enabled = compression_enabled
        self.snapshot_on_shutdown = snapshot_on_shutdown
        
        # Create standard subdirectories
        self.state_directory = os.path.join(base_directory, "state")
        self.training_directory = os.path.join(base_directory, "training")
        self.config_directory = os.path.join(base_directory, "config")
        self.logs_directory = os.path.join(base_directory, "logs")
        self.snapshots_directory = os.path.join(base_directory, "snapshots")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "base_directory": self.base_directory,
            "autosave_interval": self.autosave_interval,
            "max_snapshots": self.max_snapshots,
            "compression_enabled": self.compression_enabled,
            "snapshot_on_shutdown": self.snapshot_on_shutdown,
            "state_directory": self.state_directory,
            "training_directory": self.training_directory,
            "config_directory": self.config_directory,
            "logs_directory": self.logs_directory,
            "snapshots_directory": self.snapshots_directory
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersistenceConfig':
        """Create from dictionary representation"""
        config = cls(
            base_directory=data.get("base_directory", "data"),
            autosave_interval=data.get("autosave_interval", 300),
            max_snapshots=data.get("max_snapshots", 5),
            compression_enabled=data.get("compression_enabled", True),
            snapshot_on_shutdown=data.get("snapshot_on_shutdown", True)
        )
        
        # Override standard directories if present
        if "state_directory" in data:
            config.state_directory = data["state_directory"]
        if "training_directory" in data:
            config.training_directory = data["training_directory"]
        if "config_directory" in data:
            config.config_directory = data["config_directory"]
        if "logs_directory" in data:
            config.logs_directory = data["logs_directory"]
        if "snapshots_directory" in data:
            config.snapshots_directory = data["snapshots_directory"]
            
        return config


class PersistenceLayer:
    """
    Core persistence layer for the Lumina v1 system
    
    Handles:
    1. State storage (network topology, node states)
    2. Training data management
    3. Configuration storage
    4. Snapshot management
    5. Autosave functionality
    """
    
    def __init__(self, config: Optional[PersistenceConfig] = None):
        # Use default config if none provided
        self.config = config or PersistenceConfig()
        
        # Create directories
        self._create_directories()
        
        # Threading support
        self._lock = threading.RLock()
        
        # Autosave
        self._autosave_thread = None
        self._autosave_active = False
        
        # State caching
        self._state_cache: Dict[str, Any] = {}
        self._state_modified_flags: Dict[str, bool] = {}
        
        # Callback registry
        self._save_callbacks: Dict[str, List[callable]] = {}
        self._load_callbacks: Dict[str, List[callable]] = {}
    
    def _create_directories(self) -> None:
        """Create necessary directories for persistence"""
        directories = [
            self.config.base_directory,
            self.config.state_directory,
            self.config.training_directory,
            self.config.config_directory,
            self.config.logs_directory,
            self.config.snapshots_directory
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")
    
    # State Management
    
    def save_state(self, component_id: str, state_data: Any) -> bool:
        """
        Save state data for a component
        
        Args:
            component_id: ID of the component/node
            state_data: State to save (must be JSON serializable or picklable)
            
        Returns:
            bool: Success status
        """
        with self._lock:
            # Update state cache
            self._state_cache[component_id] = state_data
            self._state_modified_flags[component_id] = True
            
            # Construct state path
            state_file = os.path.join(self.config.state_directory, f"{component_id}.state")
            
            try:
                # Save state to file
                if isinstance(state_data, dict) or isinstance(state_data, list):
                    # Use JSON for simple data types
                    with open(state_file, 'w') as f:
                        json.dump(state_data, f, indent=2)
                else:
                    # Use pickle for complex objects
                    with open(state_file, 'wb') as f:
                        pickle.dump(state_data, f)
                
                logger.debug(f"Saved state for component: {component_id}")
                
                # Call save callbacks
                self._trigger_save_callbacks(component_id, state_data)
                
                return True
                
            except Exception as e:
                logger.error(f"Error saving state for {component_id}: {str(e)}")
                return False
    
    def load_state(self, component_id: str, default: Any = None) -> Any:
        """
        Load state data for a component
        
        Args:
            component_id: ID of the component/node
            default: Default value if no state exists
            
        Returns:
            Any: The loaded state data or default value
        """
        with self._lock:
            # Check cache first
            if component_id in self._state_cache:
                logger.debug(f"Loaded state for {component_id} from cache")
                return self._state_cache[component_id]
            
            # Construct state path
            state_file = os.path.join(self.config.state_directory, f"{component_id}.state")
            
            # Check if state file exists
            if not os.path.exists(state_file):
                logger.debug(f"No state file found for {component_id}, using default")
                self._state_cache[component_id] = default
                return default
            
            try:
                # Try to load as JSON
                try:
                    with open(state_file, 'r') as f:
                        state_data = json.load(f)
                        self._state_cache[component_id] = state_data
                        logger.debug(f"Loaded JSON state for {component_id}")
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # Try to load as pickle
                    with open(state_file, 'rb') as f:
                        state_data = pickle.load(f)
                        self._state_cache[component_id] = state_data
                        logger.debug(f"Loaded pickle state for {component_id}")
                
                # Call load callbacks
                self._trigger_load_callbacks(component_id, state_data)
                
                return state_data
                
            except Exception as e:
                logger.error(f"Error loading state for {component_id}: {str(e)}")
                self._state_cache[component_id] = default
                return default
    
    def delete_state(self, component_id: str) -> bool:
        """
        Delete state data for a component
        
        Args:
            component_id: ID of the component/node
            
        Returns:
            bool: Success status
        """
        with self._lock:
            # Remove from cache
            if component_id in self._state_cache:
                del self._state_cache[component_id]
            
            if component_id in self._state_modified_flags:
                del self._state_modified_flags[component_id]
            
            # Construct state path
            state_file = os.path.join(self.config.state_directory, f"{component_id}.state")
            
            # Delete file if it exists
            if os.path.exists(state_file):
                try:
                    os.remove(state_file)
                    logger.debug(f"Deleted state for component: {component_id}")
                    return True
                except Exception as e:
                    logger.error(f"Error deleting state for {component_id}: {str(e)}")
                    return False
            
            return True  # Nothing to delete
    
    def list_components(self) -> List[str]:
        """
        List all components with saved state
        
        Returns:
            List[str]: List of component IDs
        """
        with self._lock:
            components = set()
            
            # Add components from cache
            components.update(self._state_cache.keys())
            
            # Add components from disk
            state_files = os.listdir(self.config.state_directory)
            for state_file in state_files:
                if state_file.endswith(".state"):
                    component_id = state_file[:-6]  # Remove .state suffix
                    components.add(component_id)
            
            return sorted(list(components))
    
    # Training Data Management
    
    def save_training_data(self, dataset_name: str, data: Any) -> bool:
        """
        Save training data to disk
        
        Args:
            dataset_name: Name of the training dataset
            data: Training data to save
            
        Returns:
            bool: Success status
        """
        with self._lock:
            # Construct data path
            data_file = os.path.join(self.config.training_directory, f"{dataset_name}.data")
            
            try:
                # Save data to file based on type
                if isinstance(data, dict) or isinstance(data, list):
                    # Use JSON for simple data types
                    with open(data_file, 'w') as f:
                        json.dump(data, f, indent=2)
                else:
                    # Use pickle for complex objects
                    with open(data_file, 'wb') as f:
                        pickle.dump(data, f)
                
                logger.debug(f"Saved training data: {dataset_name}")
                return True
                
            except Exception as e:
                logger.error(f"Error saving training data {dataset_name}: {str(e)}")
                return False
    
    def load_training_data(self, dataset_name: str, default: Any = None) -> Any:
        """
        Load training data from disk
        
        Args:
            dataset_name: Name of the training dataset
            default: Default value if dataset doesn't exist
            
        Returns:
            Any: The loaded training data or default value
        """
        with self._lock:
            # Construct data path
            data_file = os.path.join(self.config.training_directory, f"{dataset_name}.data")
            
            # Check if data file exists
            if not os.path.exists(data_file):
                logger.debug(f"No training data found for {dataset_name}, using default")
                return default
            
            try:
                # Try to load as JSON
                try:
                    with open(data_file, 'r') as f:
                        data = json.load(f)
                        logger.debug(f"Loaded JSON training data: {dataset_name}")
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # Try to load as pickle
                    with open(data_file, 'rb') as f:
                        data = pickle.load(f)
                        logger.debug(f"Loaded pickle training data: {dataset_name}")
                
                return data
                
            except Exception as e:
                logger.error(f"Error loading training data {dataset_name}: {str(e)}")
                return default
    
    def list_training_datasets(self) -> List[str]:
        """
        List all available training datasets
        
        Returns:
            List[str]: List of dataset names
        """
        with self._lock:
            datasets = []
            
            # Scan directory for data files
            for file in os.listdir(self.config.training_directory):
                if file.endswith(".data"):
                    dataset_name = file[:-5]  # Remove .data suffix
                    datasets.append(dataset_name)
            
            return sorted(datasets)
    
    # Configuration Management
    
    def save_config(self, config_name: str, config_data: Dict[str, Any]) -> bool:
        """
        Save configuration data
        
        Args:
            config_name: Name of the configuration
            config_data: Configuration data to save
            
        Returns:
            bool: Success status
        """
        with self._lock:
            # Construct config path
            config_file = os.path.join(self.config.config_directory, f"{config_name}.json")
            
            try:
                # Save config to file
                with open(config_file, 'w') as f:
                    json.dump(config_data, f, indent=2)
                
                logger.debug(f"Saved configuration: {config_name}")
                return True
                
            except Exception as e:
                logger.error(f"Error saving configuration {config_name}: {str(e)}")
                return False
    
    def load_config(self, config_name: str, default: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Load configuration data
        
        Args:
            config_name: Name of the configuration
            default: Default configuration if none exists
            
        Returns:
            Dict[str, Any]: Configuration data
        """
        with self._lock:
            # Construct config path
            config_file = os.path.join(self.config.config_directory, f"{config_name}.json")
            
            # Check if config file exists
            if not os.path.exists(config_file):
                logger.debug(f"No configuration found for {config_name}, using default")
                return default if default is not None else {}
            
            try:
                # Load config from file
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                
                logger.debug(f"Loaded configuration: {config_name}")
                return config_data
                
            except Exception as e:
                logger.error(f"Error loading configuration {config_name}: {str(e)}")
                return default if default is not None else {}
    
    def list_configs(self) -> List[str]:
        """
        List all available configurations
        
        Returns:
            List[str]: List of configuration names
        """
        with self._lock:
            configs = []
            
            # Scan directory for config files
            for file in os.listdir(self.config.config_directory):
                if file.endswith(".json"):
                    config_name = file[:-5]  # Remove .json suffix
                    configs.append(config_name)
            
            return sorted(configs)
    
    # Snapshot Management
    
    def create_snapshot(self, description: str = "") -> str:
        """
        Create a snapshot of the current state and configuration
        
        Args:
            description: Optional description of the snapshot
            
        Returns:
            str: Snapshot ID
        """
        with self._lock:
            # Ensure all modified states are saved
            self._flush_cache()
            
            # Generate snapshot ID using timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_id = f"snapshot_{timestamp}"
            
            # Create snapshot directory
            snapshot_dir = os.path.join(self.config.snapshots_directory, snapshot_id)
            os.makedirs(snapshot_dir, exist_ok=True)
            
            # Copy state files
            state_dir = os.path.join(snapshot_dir, "state")
            os.makedirs(state_dir, exist_ok=True)
            for file in os.listdir(self.config.state_directory):
                src = os.path.join(self.config.state_directory, file)
                dst = os.path.join(state_dir, file)
                shutil.copy2(src, dst)
            
            # Copy config files
            config_dir = os.path.join(snapshot_dir, "config")
            os.makedirs(config_dir, exist_ok=True)
            for file in os.listdir(self.config.config_directory):
                src = os.path.join(self.config.config_directory, file)
                dst = os.path.join(config_dir, file)
                shutil.copy2(src, dst)
            
            # Create snapshot metadata
            metadata = {
                "snapshot_id": snapshot_id,
                "timestamp": timestamp,
                "description": description,
                "components": self.list_components(),
                "configs": self.list_configs()
            }
            
            # Save metadata
            metadata_file = os.path.join(snapshot_dir, "metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Created snapshot: {snapshot_id}")
            
            # Manage maximum number of snapshots
            self._enforce_max_snapshots()
            
            return snapshot_id
    
    def restore_snapshot(self, snapshot_id: str) -> bool:
        """
        Restore the system from a snapshot
        
        Args:
            snapshot_id: ID of the snapshot to restore
            
        Returns:
            bool: Success status
        """
        with self._lock:
            # Validate snapshot exists
            snapshot_dir = os.path.join(self.config.snapshots_directory, snapshot_id)
            if not os.path.exists(snapshot_dir):
                logger.error(f"Snapshot {snapshot_id} not found")
                return False
            
            try:
                # Clear state cache
                self._state_cache.clear()
                self._state_modified_flags.clear()
                
                # Clear current state directory
                for file in os.listdir(self.config.state_directory):
                    os.remove(os.path.join(self.config.state_directory, file))
                
                # Clear current config directory
                for file in os.listdir(self.config.config_directory):
                    os.remove(os.path.join(self.config.config_directory, file))
                
                # Copy state files from snapshot
                snapshot_state_dir = os.path.join(snapshot_dir, "state")
                for file in os.listdir(snapshot_state_dir):
                    src = os.path.join(snapshot_state_dir, file)
                    dst = os.path.join(self.config.state_directory, file)
                    shutil.copy2(src, dst)
                
                # Copy config files from snapshot
                snapshot_config_dir = os.path.join(snapshot_dir, "config")
                for file in os.listdir(snapshot_config_dir):
                    src = os.path.join(snapshot_config_dir, file)
                    dst = os.path.join(self.config.config_directory, file)
                    shutil.copy2(src, dst)
                
                logger.info(f"Restored snapshot: {snapshot_id}")
                return True
                
            except Exception as e:
                logger.error(f"Error restoring snapshot {snapshot_id}: {str(e)}")
                return False
    
    def list_snapshots(self) -> List[Dict[str, Any]]:
        """
        List all available snapshots with metadata
        
        Returns:
            List[Dict[str, Any]]: List of snapshot metadata
        """
        with self._lock:
            snapshots = []
            
            # Scan snapshots directory
            for item in os.listdir(self.config.snapshots_directory):
                snapshot_dir = os.path.join(self.config.snapshots_directory, item)
                if os.path.isdir(snapshot_dir):
                    # Load metadata
                    metadata_file = os.path.join(snapshot_dir, "metadata.json")
                    if os.path.exists(metadata_file):
                        try:
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                                snapshots.append(metadata)
                        except Exception as e:
                            logger.warning(f"Error loading metadata for snapshot {item}: {str(e)}")
            
            # Sort by timestamp (most recent first)
            snapshots.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            return snapshots
    
    def _enforce_max_snapshots(self) -> None:
        """Enforce maximum number of snapshots by removing oldest"""
        snapshots = self.list_snapshots()
        
        # Remove oldest snapshots if we exceed the maximum
        while len(snapshots) > self.config.max_snapshots:
            oldest = snapshots.pop()
            oldest_id = oldest.get("snapshot_id")
            if oldest_id:
                oldest_dir = os.path.join(self.config.snapshots_directory, oldest_id)
                try:
                    shutil.rmtree(oldest_dir)
                    logger.debug(f"Removed oldest snapshot: {oldest_id}")
                except Exception as e:
                    logger.error(f"Error removing snapshot {oldest_id}: {str(e)}")
    
    # Autosave Functionality
    
    def start_autosave(self) -> None:
        """Start the autosave thread"""
        with self._lock:
            if self._autosave_thread is not None and self._autosave_thread.is_alive():
                logger.warning("Autosave thread already running")
                return
            
            self._autosave_active = True
            self._autosave_thread = threading.Thread(target=self._autosave_loop)
            self._autosave_thread.daemon = True
            self._autosave_thread.start()
            logger.info("Started autosave thread")
    
    def stop_autosave(self) -> None:
        """Stop the autosave thread"""
        with self._lock:
            self._autosave_active = False
            if self._autosave_thread:
                self._autosave_thread.join(timeout=2.0)
                logger.info("Stopped autosave thread")
    
    def _autosave_loop(self) -> None:
        """Autosave loop that runs in a separate thread"""
        while self._autosave_active:
            try:
                # Sleep for the interval (but check periodically if we should stop)
                for _ in range(int(self.config.autosave_interval * 10)):
                    if not self._autosave_active:
                        break
                    time.sleep(0.1)
                
                # Skip if not active anymore
                if not self._autosave_active:
                    break
                
                # Flush modified states to disk
                saved_count = self._flush_cache()
                
                if saved_count > 0:
                    logger.debug(f"Autosaved {saved_count} modified states")
                
            except Exception as e:
                logger.error(f"Error in autosave thread: {str(e)}")
    
    def _flush_cache(self) -> int:
        """
        Flush all modified states to disk
        
        Returns:
            int: Number of states saved
        """
        with self._lock:
            saved_count = 0
            
            # Save all modified states
            for component_id, modified in list(self._state_modified_flags.items()):
                if modified and component_id in self._state_cache:
                    # Construct state path
                    state_file = os.path.join(self.config.state_directory, f"{component_id}.state")
                    
                    try:
                        # Save state to file
                        state_data = self._state_cache[component_id]
                        if isinstance(state_data, dict) or isinstance(state_data, list):
                            # Use JSON for simple data types
                            with open(state_file, 'w') as f:
                                json.dump(state_data, f, indent=2)
                        else:
                            # Use pickle for complex objects
                            with open(state_file, 'wb') as f:
                                pickle.dump(state_data, f)
                        
                        # Mark as not modified
                        self._state_modified_flags[component_id] = False
                        saved_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error flushing state for {component_id}: {str(e)}")
            
            return saved_count
    
    # Callback Registration
    
    def register_save_callback(self, component_id: str, callback: callable) -> None:
        """
        Register a callback to be called after state is saved
        
        Args:
            component_id: ID of the component/node
            callback: Function to call with (component_id, state_data)
        """
        with self._lock:
            if component_id not in self._save_callbacks:
                self._save_callbacks[component_id] = []
            
            self._save_callbacks[component_id].append(callback)
    
    def register_load_callback(self, component_id: str, callback: callable) -> None:
        """
        Register a callback to be called after state is loaded
        
        Args:
            component_id: ID of the component/node
            callback: Function to call with (component_id, state_data)
        """
        with self._lock:
            if component_id not in self._load_callbacks:
                self._load_callbacks[component_id] = []
            
            self._load_callbacks[component_id].append(callback)
    
    def _trigger_save_callbacks(self, component_id: str, state_data: Any) -> None:
        """Trigger all save callbacks for a component"""
        callbacks = self._save_callbacks.get(component_id, [])
        for callback in callbacks:
            try:
                callback(component_id, state_data)
            except Exception as e:
                logger.error(f"Error in save callback for {component_id}: {str(e)}")
    
    def _trigger_load_callbacks(self, component_id: str, state_data: Any) -> None:
        """Trigger all load callbacks for a component"""
        callbacks = self._load_callbacks.get(component_id, [])
        for callback in callbacks:
            try:
                callback(component_id, state_data)
            except Exception as e:
                logger.error(f"Error in load callback for {component_id}: {str(e)}")
    
    # Cleanup
    
    def shutdown(self) -> None:
        """Shutdown the persistence layer"""
        logger.info("Shutting down persistence layer")
        
        # Stop autosave
        self.stop_autosave()
        
        # Flush modified states
        self._flush_cache()
        
        # Create shutdown snapshot if configured
        if self.config.snapshot_on_shutdown:
            self.create_snapshot("Automatic shutdown snapshot") 