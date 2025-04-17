"""
V7 Socket Manager

Extends the V5 socket architecture with enhanced capabilities for knowledge
representation and autonomous learning processes.
"""

import logging
import json
import time
from pathlib import Path
import sys
from datetime import datetime

# Add project root to path if needed
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import V5 socket manager
try:
    from src.v5.socket_manager import SocketManager as V5SocketManager
except ImportError:
    logging.warning("V5 SocketManager not found. Using mock implementation.")
    # Create a minimal mock implementation for development
    class V5SocketManager:
        def __init__(self):
            self.plugins = {}
            self.message_queue = []
            self.subscribers = {}
            
        def register_plugin(self, plugin):
            plugin_id = getattr(plugin, 'plugin_id', str(id(plugin)))
            self.plugins[plugin_id] = plugin
            
        def send_message(self, message):
            self.message_queue.append(message)
            self._process_message(message)
            
        def _process_message(self, message):
            message_type = message.get('type', '')
            if message_type in self.subscribers:
                for callback in self.subscribers[message_type]:
                    callback(message)
                    
        def subscribe(self, message_type, callback):
            if message_type not in self.subscribers:
                self.subscribers[message_type] = []
            self.subscribers[message_type].append(callback)

# Set up logging
logger = logging.getLogger(__name__)

class V7SocketManager(V5SocketManager):
    """
    Enhanced socket manager for V7 components with knowledge representation capabilities.
    
    Extends the V5 socket architecture to handle knowledge graphs, learning pathways,
    and autonomous learning processes.
    """
    
    def __init__(self):
        """Initialize the V7 socket manager"""
        super().__init__()
        
        # Initialize specialized plugin registries
        self.knowledge_plugins = {}  # Knowledge-specific plugins
        self.learning_controllers = {}  # Learning control components
        self.auto_wiki_plugins = {}  # AutoWiki integration plugins
        
        # Enhanced message handlers
        self.knowledge_handlers = {
            "knowledge_update": self._handle_knowledge_update,
            "graph_change": self._handle_graph_change,
            "learning_event": self._handle_learning_event,
            "domain_query": self._handle_domain_query,
            "learning_control": self._handle_learning_control
        }
        
        # Knowledge graph cache
        self.graph_cache = {}
        self.cache_timestamp = {}
        self.cache_ttl = 60  # Cache time-to-live in seconds
        
        logger.info("V7 Socket Manager initialized")
    
    def register_knowledge_plugin(self, plugin):
        """
        Register a knowledge-specific plugin
        
        Args:
            plugin: The knowledge plugin to register
        """
        # Get plugin ID
        plugin_id = self._get_plugin_id(plugin)
        
        # Register with specialized registry
        self.knowledge_plugins[plugin_id] = plugin
        
        # Also register with standard plugin system
        self.register_plugin(plugin)
        
        # Extract supported domains if available
        if hasattr(plugin, 'get_supported_domains'):
            domains = plugin.get_supported_domains()
            logger.info(f"Registered knowledge plugin {plugin_id} with domains: {domains}")
        else:
            logger.info(f"Registered knowledge plugin {plugin_id}")
    
    def register_learning_controller(self, controller):
        """
        Register a learning controller component
        
        Args:
            controller: The learning controller to register
        """
        # Get controller ID
        controller_id = self._get_controller_id(controller)
        
        # Register with specialized registry
        self.learning_controllers[controller_id] = controller
        
        logger.info(f"Registered learning controller {controller_id}")
    
    def register_auto_wiki_plugin(self, plugin):
        """
        Register an AutoWiki integration plugin
        
        Args:
            plugin: The AutoWiki plugin to register
        """
        # Get plugin ID
        plugin_id = self._get_plugin_id(plugin)
        
        # Register with specialized registry
        self.auto_wiki_plugins[plugin_id] = plugin
        
        # Also register with standard plugin system
        self.register_plugin(plugin)
        
        logger.info(f"Registered AutoWiki plugin {plugin_id}")
    
    def get_knowledge_graph(self, domain=None, depth=2, use_cache=True):
        """
        Retrieve knowledge graph data from registered plugins
        
        Args:
            domain: Optional domain filter
            depth: Maximum graph depth
            use_cache: Whether to use cached data if available
            
        Returns:
            dict: Combined knowledge graph data
        """
        # Check cache if enabled
        cache_key = f"{domain}_{depth}"
        if use_cache and cache_key in self.graph_cache:
            cache_age = time.time() - self.cache_timestamp.get(cache_key, 0)
            if cache_age < self.cache_ttl:
                logger.debug(f"Using cached knowledge graph for {cache_key}")
                return self.graph_cache[cache_key]
        
        # Initialize empty graph
        graph_data = {"nodes": [], "edges": []}
        
        # Collect data from all knowledge plugins
        for plugin_id, plugin in self.knowledge_plugins.items():
            if hasattr(plugin, 'get_knowledge_graph'):
                try:
                    plugin_data = plugin.get_knowledge_graph(domain, depth)
                    self._merge_graph_data(graph_data, plugin_data)
                    logger.debug(f"Added graph data from plugin {plugin_id}")
                except Exception as e:
                    logger.error(f"Error getting knowledge graph from plugin {plugin_id}: {e}")
        
        # Update cache
        self.graph_cache[cache_key] = graph_data
        self.cache_timestamp[cache_key] = time.time()
        
        return graph_data
    
    def get_learning_pathway(self, topic=None, timeframe=None):
        """
        Retrieve learning pathway data for a topic
        
        Args:
            topic: Topic identifier
            timeframe: Optional timeframe filter (e.g., "24h", "7d", "30d")
            
        Returns:
            dict: Learning pathway data including nodes, connections, and decision points
        """
        pathway_data = {
            "topic": topic,
            "timeframe": timeframe,
            "nodes": [],
            "connections": [],
            "decision_points": []
        }
        
        # Collect data from knowledge plugins
        for plugin_id, plugin in self.knowledge_plugins.items():
            if hasattr(plugin, 'get_learning_pathway'):
                try:
                    plugin_data = plugin.get_learning_pathway(topic, timeframe)
                    self._merge_pathway_data(pathway_data, plugin_data)
                    logger.debug(f"Added pathway data from plugin {plugin_id}")
                except Exception as e:
                    logger.error(f"Error getting learning pathway from plugin {plugin_id}: {e}")
        
        return pathway_data
    
    def get_auto_wiki_status(self):
        """
        Get the current status of the AutoWiki system
        
        Returns:
            dict: Status information for the AutoWiki system
        """
        status = {
            "active": False,
            "queue_size": 0,
            "recent_acquisitions": [],
            "verification_status": {},
            "integration_status": {}
        }
        
        # Check if any AutoWiki plugins are registered
        if not self.auto_wiki_plugins:
            logger.warning("No AutoWiki plugins registered")
            return status
        
        # Get status from all AutoWiki plugins
        for plugin_id, plugin in self.auto_wiki_plugins.items():
            if hasattr(plugin, 'get_status'):
                try:
                    plugin_status = plugin.get_status()
                    self._merge_status_data(status, plugin_status)
                    logger.debug(f"Added status data from plugin {plugin_id}")
                except Exception as e:
                    logger.error(f"Error getting AutoWiki status from plugin {plugin_id}: {e}")
        
        return status
    
    def set_learning_parameters(self, parameters):
        """
        Set learning parameters for all learning controllers
        
        Args:
            parameters: Dictionary of learning parameters
            
        Returns:
            dict: Status of parameter updates
        """
        results = {}
        
        # Update parameters on all learning controllers
        for controller_id, controller in self.learning_controllers.items():
            if hasattr(controller, 'set_parameters'):
                try:
                    success = controller.set_parameters(parameters)
                    results[controller_id] = {"success": success}
                    logger.debug(f"Updated parameters for controller {controller_id}")
                except Exception as e:
                    results[controller_id] = {"success": False, "error": str(e)}
                    logger.error(f"Error setting parameters for controller {controller_id}: {e}")
        
        # Broadcast parameter change to all interested plugins
        self.send_message({
            "type": "learning_parameters_changed",
            "plugin_id": "v7_socket_manager",
            "timestamp": datetime.now().isoformat(),
            "content": {
                "parameters": parameters,
                "results": results
            }
        })
        
        return results
    
    def send_knowledge_update(self, domain, topic, operation, data):
        """
        Send a knowledge update message to all subscribers
        
        Args:
            domain: Knowledge domain
            topic: Topic identifier
            operation: Operation type (add_node, update_node, add_edge, etc.)
            data: Operation-specific data
            
        Returns:
            str: Message ID
        """
        message_id = f"knowledge_update_{int(time.time() * 1000)}"
        
        message = {
            "type": "knowledge_update",
            "plugin_id": "v7_socket_manager",
            "request_id": message_id,
            "timestamp": datetime.now().isoformat(),
            "content": {
                "domain": domain,
                "topic": topic,
                "operation": operation,
                "data": data
            }
        }
        
        # Send message through the socket system
        self.send_message(message)
        
        return message_id
    
    def _merge_graph_data(self, target, source):
        """
        Merge knowledge graph data from multiple sources
        
        Args:
            target: Target graph dict to merge into
            source: Source graph dict to merge from
        """
        if not isinstance(source, dict):
            logger.warning(f"Invalid graph data format: {type(source)}")
            return
        
        # Extract nodes and edges
        source_nodes = source.get("nodes", [])
        source_edges = source.get("edges", [])
        
        # Node merging with deduplication by ID
        existing_ids = {node['id'] for node in target['nodes']}
        for node in source_nodes:
            if 'id' not in node:
                logger.warning(f"Node missing ID field: {node}")
                continue
                
            if node['id'] not in existing_ids:
                target['nodes'].append(node)
                existing_ids.add(node['id'])
            else:
                # Merge properties for existing nodes
                for existing_node in target['nodes']:
                    if existing_node['id'] == node['id']:
                        # Merge properties, with target taking precedence
                        for key, value in node.items():
                            if key not in existing_node:
                                existing_node[key] = value
                        break
        
        # Edge merging with deduplication
        existing_edges = {(edge['source'], edge['target']) for edge in target['edges']}
        for edge in source_edges:
            if 'source' not in edge or 'target' not in edge:
                logger.warning(f"Edge missing source or target: {edge}")
                continue
                
            edge_key = (edge['source'], edge['target'])
            if edge_key not in existing_edges:
                target['edges'].append(edge)
                existing_edges.add(edge_key)
            else:
                # Merge properties for existing edges
                for existing_edge in target['edges']:
                    if existing_edge['source'] == edge['source'] and existing_edge['target'] == edge['target']:
                        # Merge properties, with target taking precedence
                        for key, value in edge.items():
                            if key not in existing_edge:
                                existing_edge[key] = value
                        break
    
    def _merge_pathway_data(self, target, source):
        """
        Merge learning pathway data from multiple sources
        
        Args:
            target: Target pathway dict to merge into
            source: Source pathway dict to merge from
        """
        if not isinstance(source, dict):
            logger.warning(f"Invalid pathway data format: {type(source)}")
            return
        
        # Extract nodes, connections, and decision points
        source_nodes = source.get("nodes", [])
        source_connections = source.get("connections", [])
        source_decision_points = source.get("decision_points", [])
        
        # Node merging with deduplication
        existing_nodes = {node['id'] for node in target['nodes']}
        for node in source_nodes:
            if 'id' not in node:
                logger.warning(f"Pathway node missing ID field: {node}")
                continue
                
            if node['id'] not in existing_nodes:
                target['nodes'].append(node)
                existing_nodes.add(node['id'])
        
        # Connection merging with deduplication
        existing_connections = {(conn['source'], conn['target']) for conn in target['connections']}
        for conn in source_connections:
            if 'source' not in conn or 'target' not in conn:
                logger.warning(f"Connection missing source or target: {conn}")
                continue
                
            conn_key = (conn['source'], conn['target'])
            if conn_key not in existing_connections:
                target['connections'].append(conn)
                existing_connections.add(conn_key)
        
        # Decision point merging with deduplication
        existing_decisions = {dec['id'] for dec in target['decision_points']}
        for dec in source_decision_points:
            if 'id' not in dec:
                logger.warning(f"Decision point missing ID field: {dec}")
                continue
                
            if dec['id'] not in existing_decisions:
                target['decision_points'].append(dec)
                existing_decisions.add(dec['id'])
    
    def _merge_status_data(self, target, source):
        """
        Merge AutoWiki status data from multiple sources
        
        Args:
            target: Target status dict to merge into
            source: Source status dict to merge from
        """
        if not isinstance(source, dict):
            logger.warning(f"Invalid status data format: {type(source)}")
            return
        
        # Merge active status (true if any source is active)
        if source.get("active", False):
            target["active"] = True
        
        # Sum queue sizes
        target["queue_size"] += source.get("queue_size", 0)
        
        # Merge recent acquisitions
        source_acquisitions = source.get("recent_acquisitions", [])
        existing_acquisitions = {acq['id'] for acq in target['recent_acquisitions']}
        for acq in source_acquisitions:
            if 'id' not in acq:
                logger.warning(f"Acquisition missing ID field: {acq}")
                continue
                
            if acq['id'] not in existing_acquisitions:
                target['recent_acquisitions'].append(acq)
                existing_acquisitions.add(acq['id'])
        
        # Sort acquisitions by timestamp (newest first)
        target['recent_acquisitions'].sort(
            key=lambda x: x.get('timestamp', ''), 
            reverse=True
        )
        
        # Limit to 10 most recent
        target['recent_acquisitions'] = target['recent_acquisitions'][:10]
        
        # Merge verification status
        for topic, status in source.get("verification_status", {}).items():
            if topic not in target["verification_status"]:
                target["verification_status"][topic] = status
        
        # Merge integration status
        for topic, status in source.get("integration_status", {}).items():
            if topic not in target["integration_status"]:
                target["integration_status"][topic] = status
    
    def _get_plugin_id(self, plugin):
        """
        Get the plugin ID in a consistent way
        
        Args:
            plugin: Plugin object
            
        Returns:
            str: Plugin ID
        """
        if hasattr(plugin, 'get_plugin_id'):
            return plugin.get_plugin_id()
        elif hasattr(plugin, 'plugin_id'):
            return plugin.plugin_id
        else:
            # Generate a unique ID based on object id
            return f"plugin_{id(plugin)}"
    
    def _get_controller_id(self, controller):
        """
        Get the controller ID in a consistent way
        
        Args:
            controller: Controller object
            
        Returns:
            str: Controller ID
        """
        if hasattr(controller, 'get_controller_id'):
            return controller.get_controller_id()
        elif hasattr(controller, 'controller_id'):
            return controller.controller_id
        else:
            # Generate a unique ID based on object id
            return f"controller_{id(controller)}"
    
    def _handle_knowledge_update(self, message):
        """
        Handle knowledge update messages
        
        Args:
            message: Knowledge update message
        """
        logger.debug(f"Handling knowledge update: {message.get('content', {}).get('operation')}")
        
        # Invalidate cache for affected domain
        domain = message.get('content', {}).get('domain')
        if domain:
            for key in list(self.cache_timestamp.keys()):
                if key.startswith(f"{domain}_"):
                    logger.debug(f"Invalidating cache for {key}")
                    self.cache_timestamp.pop(key, None)
                    self.graph_cache.pop(key, None)
    
    def _handle_graph_change(self, message):
        """
        Handle graph change messages
        
        Args:
            message: Graph change message
        """
        logger.debug("Handling graph change message")
        
        # Invalidate all graph caches
        self.graph_cache.clear()
        self.cache_timestamp.clear()
    
    def _handle_learning_event(self, message):
        """
        Handle learning event messages
        
        Args:
            message: Learning event message
        """
        logger.debug(f"Handling learning event: {message.get('content', {}).get('event_type')}")
        
        # No specific action needed - subscribers will handle this
        pass
    
    def _handle_domain_query(self, message):
        """
        Handle domain query messages
        
        Args:
            message: Domain query message
        """
        logger.debug("Handling domain query message")
        
        # Extract query parameters
        content = message.get('content', {})
        domain = content.get('domain')
        depth = content.get('depth', 2)
        use_cache = content.get('use_cache', True)
        
        # Get matching graph data
        try:
            graph_data = self.get_knowledge_graph(domain, depth, use_cache)
            
            # Send response message
            response = {
                "type": "domain_query_response",
                "plugin_id": "v7_socket_manager",
                "request_id": message.get('request_id'),
                "timestamp": datetime.now().isoformat(),
                "content": {
                    "domain": domain,
                    "depth": depth,
                    "graph_data": graph_data
                }
            }
            self.send_message(response)
            
        except Exception as e:
            logger.error(f"Error handling domain query: {e}")
            
            # Send error response
            error_response = {
                "type": "domain_query_response",
                "plugin_id": "v7_socket_manager",
                "request_id": message.get('request_id'),
                "timestamp": datetime.now().isoformat(),
                "content": {
                    "domain": domain,
                    "depth": depth,
                    "error": str(e)
                }
            }
            self.send_message(error_response)
    
    def _handle_learning_control(self, message):
        """
        Handle learning control messages
        
        Args:
            message: Learning control message
        """
        logger.debug("Handling learning control message")
        
        # Extract control parameters
        content = message.get('content', {})
        parameters = content.get('parameters', {})
        
        # Apply parameters to learning controllers
        try:
            results = self.set_learning_parameters(parameters)
            
            # Send response message
            response = {
                "type": "learning_control_response",
                "plugin_id": "v7_socket_manager",
                "request_id": message.get('request_id'),
                "timestamp": datetime.now().isoformat(),
                "content": {
                    "results": results
                }
            }
            self.send_message(response)
            
        except Exception as e:
            logger.error(f"Error handling learning control: {e}")
            
            # Send error response
            error_response = {
                "type": "learning_control_response",
                "plugin_id": "v7_socket_manager",
                "request_id": message.get('request_id'),
                "timestamp": datetime.now().isoformat(),
                "content": {
                    "error": str(e)
                }
            }
            self.send_message(error_response)


# Create a mock plugin class for testing
class MockKnowledgePlugin:
    """Mock knowledge plugin for testing V7 socket manager"""
    
    def __init__(self, plugin_id="mock_knowledge_plugin"):
        self.plugin_id = plugin_id
        
    def get_plugin_id(self):
        return self.plugin_id
        
    def get_supported_domains(self):
        return ["general", "science", "mathematics"]
        
    def get_knowledge_graph(self, domain=None, depth=2):
        """Return a mock knowledge graph for testing"""
        graph = {
            "nodes": [
                {"id": "concept1", "label": "First Concept", "domain": "general"},
                {"id": "concept2", "label": "Second Concept", "domain": "general"},
                {"id": "concept3", "label": "Third Concept", "domain": "science"}
            ],
            "edges": [
                {"source": "concept1", "target": "concept2", "type": "related"},
                {"source": "concept2", "target": "concept3", "type": "depends_on"}
            ]
        }
        
        # Filter by domain if specified
        if domain:
            graph["nodes"] = [node for node in graph["nodes"] if node.get("domain") == domain]
            
            # Only keep edges where both source and target are still in the graph
            node_ids = {node["id"] for node in graph["nodes"]}
            graph["edges"] = [
                edge for edge in graph["edges"] 
                if edge["source"] in node_ids and edge["target"] in node_ids
            ]
        
        return graph
        
    def get_learning_pathway(self, topic=None, timeframe=None):
        """Return a mock learning pathway for testing"""
        return {
            "nodes": [
                {"id": "start", "label": "Starting Point", "type": "concept"},
                {"id": "mid1", "label": "Intermediate 1", "type": "concept"},
                {"id": "mid2", "label": "Intermediate 2", "type": "concept"},
                {"id": "end", "label": "Final Concept", "type": "concept"}
            ],
            "connections": [
                {"source": "start", "target": "mid1", "type": "progress"},
                {"source": "mid1", "target": "mid2", "type": "progress"},
                {"source": "mid2", "target": "end", "type": "progress"}
            ],
            "decision_points": [
                {
                    "id": "decision1",
                    "node_id": "mid1",
                    "options": ["mid2", "alternate"],
                    "selected": "mid2",
                    "rationale": "Better compatibility with existing knowledge"
                }
            ]
        }


# Test function
def test_v7_socket_manager():
    """Test the V7 socket manager functionality"""
    socket_manager = V7SocketManager()
    
    # Create and register a mock plugin
    mock_plugin = MockKnowledgePlugin()
    socket_manager.register_knowledge_plugin(mock_plugin)
    
    # Test getting knowledge graph
    graph = socket_manager.get_knowledge_graph()
    print(f"Knowledge graph has {len(graph['nodes'])} nodes and {len(graph['edges'])} edges")
    
    # Test getting learning pathway
    pathway = socket_manager.get_learning_pathway("test_topic")
    print(f"Learning pathway has {len(pathway['nodes'])} nodes and {len(pathway['decision_points'])} decision points")
    
    # Test sending a knowledge update
    message_id = socket_manager.send_knowledge_update(
        domain="science",
        topic="physics",
        operation="add_node",
        data={"id": "new_concept", "label": "New Physics Concept"}
    )
    print(f"Sent knowledge update with message ID: {message_id}")
    
    print("V7 socket manager test completed successfully")


# Run the test if executed directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_v7_socket_manager() 