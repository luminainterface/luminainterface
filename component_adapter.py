import logging
import importlib
import json
from typing import Dict, Any, List, Callable, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("component_adapter")

class ComponentAdapter:
    """
    Adapter for integrating components with the Lumina system
    
    This class provides a standardized interface for connecting 
    different components to the core system and facilitating 
    communication between them.
    """
    
    def __init__(self, config_path: str = "config/components.json"):
        """
        Initialize the component adapter
        
        Args:
            config_path: Path to component configuration file
        """
        self.config_path = Path(config_path)
        self.components = {}
        self.interfaces = {}
        self.responders = {}
        self.entities = {}
        self.event_handlers = {}
        
        # Load configuration if available
        self.config = self._load_config()
        
        logger.info(f"ComponentAdapter initialized with {len(self.config.get('components', []))} components defined")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load component configuration"""
        default_config = {
            "components": [],
            "auto_load": True,
            "data_path": "data",
            "debug": False
        }
        
        if not self.config_path.exists():
            logger.warning(f"Configuration file not found: {self.config_path}")
            return default_config
            
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return {**default_config, **config}
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return default_config
    
    def discover_components(self) -> List[str]:
        """
        Discover available components in the system
        
        Returns:
            List of component names
        """
        discovered = []
        
        # Check for Monday reflector
        try:
            importlib.import_module("monday_reflector")
            discovered.append("monday")
            logger.info("Discovered Monday Reflector component")
        except ImportError:
            logger.debug("Monday Reflector not available")
        
        # Check for other components defined in configuration
        for component in self.config.get("components", []):
            try:
                if "module" in component:
                    importlib.import_module(component["module"])
                    discovered.append(component["name"])
                    logger.info(f"Discovered component: {component['name']}")
            except ImportError:
                logger.debug(f"Component {component.get('name')} not available")
        
        return discovered
    
    def load_component(self, component_name: str) -> bool:
        """
        Load a component by name
        
        Args:
            component_name: Name of the component to load
            
        Returns:
            True if component was loaded successfully
        """
        if component_name in self.components:
            logger.warning(f"Component {component_name} already loaded")
            return True
            
        # Handle Monday Reflector
        if component_name.lower() == "monday":
            try:
                from monday_reflector import MondayReflector
                monday = MondayReflector()
                self.components["monday"] = monday
                
                # Register Monday's responder
                self.register_responder("monday", monday.get_response)
                
                # Register Monday's entity information
                self.register_entity("monday", monday.get_core_identity())
                
                logger.info("Loaded Monday Reflector component")
                return True
            except Exception as e:
                logger.error(f"Error loading Monday Reflector: {str(e)}")
                return False
        
        # Handle other components from configuration
        for component in self.config.get("components", []):
            if component.get("name") == component_name:
                try:
                    if "module" in component and "class" in component:
                        module = importlib.import_module(component["module"])
                        component_class = getattr(module, component["class"])
                        component_instance = component_class()
                        self.components[component_name] = component_instance
                        
                        # Register interfaces if available
                        if hasattr(component_instance, "get_interface"):
                            self.interfaces[component_name] = component_instance.get_interface()
                            
                        logger.info(f"Loaded component: {component_name}")
                        return True
                except Exception as e:
                    logger.error(f"Error loading component {component_name}: {str(e)}")
                    return False
        
        logger.warning(f"Component {component_name} not found in configuration")
        return False
    
    def get_component(self, component_name: str) -> Any:
        """
        Get a loaded component by name
        
        Args:
            component_name: Name of the component
            
        Returns:
            Component instance or None if not loaded
        """
        return self.components.get(component_name)
    
    def register_responder(self, name: str, responder_func: Callable[[str], str]) -> None:
        """
        Register a response function for a component
        
        Args:
            name: Responder name
            responder_func: Function that takes a string and returns a string response
        """
        self.responders[name] = responder_func
        logger.info(f"Registered responder: {name}")
    
    def get_response(self, responder_name: str, query: str) -> Optional[str]:
        """
        Get a response from a registered responder
        
        Args:
            responder_name: Name of the responder
            query: Query string
            
        Returns:
            Response string or None if responder not found
        """
        if responder_name in self.responders:
            try:
                return self.responders[responder_name](query)
            except Exception as e:
                logger.error(f"Error getting response from {responder_name}: {str(e)}")
                return f"Error: {str(e)}"
        return None
    
    def register_entity(self, name: str, entity_data: Dict[str, Any]) -> None:
        """
        Register entity information
        
        Args:
            name: Entity name
            entity_data: Dictionary of entity data
        """
        self.entities[name] = entity_data
        logger.info(f"Registered entity: {name}")
    
    def get_entity(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get entity information by name
        
        Args:
            name: Entity name
            
        Returns:
            Entity data or None if not found
        """
        return self.entities.get(name)
    
    def register_event_handler(self, event_type: str, handler_func: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register an event handler
        
        Args:
            event_type: Type of event to handle
            handler_func: Function to call when event occurs
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
            
        self.event_handlers[event_type].append(handler_func)
        logger.info(f"Registered event handler for: {event_type}")
    
    def trigger_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Trigger an event
        
        Args:
            event_type: Type of event
            event_data: Event data
        """
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    handler(event_data)
                except Exception as e:
                    logger.error(f"Error in event handler for {event_type}: {str(e)}")
                    
        logger.debug(f"Triggered event: {event_type}")

    def integrate_with_lumina(self, lumina_system) -> bool:
        """
        Integrate components with the Lumina system
        
        Args:
            lumina_system: Lumina system instance
            
        Returns:
            True if integration was successful
        """
        success = True
        
        # Integrate Monday if available
        if "monday" in self.components:
            try:
                monday = self.components["monday"]
                if hasattr(monday, "integrate_with_lumina"):
                    monday_success = monday.integrate_with_lumina(lumina_system)
                    if not monday_success:
                        logger.warning("Monday integration with Lumina failed")
                        success = False
                else:
                    # Fallback integration
                    lumina_system.register_entity(monday.get_core_identity())
                    lumina_system.register_responder("monday", monday.get_response)
                    logger.info("Completed fallback Monday integration with Lumina")
            except Exception as e:
                logger.error(f"Error integrating Monday with Lumina: {str(e)}")
                success = False
        
        # Integrate other components
        for name, component in self.components.items():
            if name == "monday":
                continue  # Already handled
                
            try:
                if hasattr(component, "integrate_with_lumina"):
                    component_success = component.integrate_with_lumina(lumina_system)
                    if not component_success:
                        logger.warning(f"{name} integration with Lumina failed")
                        success = False
            except Exception as e:
                logger.error(f"Error integrating {name} with Lumina: {str(e)}")
                success = False
        
        return success

def create_adapter() -> ComponentAdapter:
    """
    Create and initialize a component adapter
    
    Returns:
        Initialized ComponentAdapter instance
    """
    adapter = ComponentAdapter()
    
    # Discover and load components if auto_load is enabled
    if adapter.config.get("auto_load", True):
        components = adapter.discover_components()
        for component in components:
            adapter.load_component(component)
    
    return adapter 