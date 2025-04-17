"""
API Service Plugin for V5 Visualization System

This module provides a REST API service for the V5 visualization system, allowing 
external applications to access neural state, pattern processing, and consciousness
metrics data.
"""

import json
import logging
import threading
import uuid
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs

from .node_socket import NodeSocket

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import the connection discovery service
try:
    from connection_discovery import register_node
    HAS_DISCOVERY = True
except ImportError:
    logger.warning("ConnectionDiscovery not available. Limited API service discovery will be used.")
    HAS_DISCOVERY = False
    # Define a dummy register_node function
    def register_node(node, **kwargs):
        logger.warning(f"Cannot register node: {node.node_id}")
        return None


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """HTTP Server that handles requests in separate threads"""
    pass


class APIRequestHandler(BaseHTTPRequestHandler):
    """Handler for API requests"""
    
    def __init__(self, *args, plugin=None, **kwargs):
        self.plugin = plugin
        super().__init__(*args, **kwargs)
    
    def log_message(self, format, *args):
        """Override to use our logger instead of stderr"""
        logger.debug(format % args)
    
    def _set_headers(self, status_code=200, content_type='application/json'):
        """Set common response headers"""
        self.send_response(status_code)
        self.send_header('Content-type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS"""
        self._set_headers()
    
    def do_GET(self):
        """Handle GET requests"""
        if not self.plugin:
            self._send_error(500, "Server configuration error")
            return
        
        # Parse URL and query parameters
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query = parse_qs(parsed_url.query)
        
        try:
            # Routes for neural state
            if path == '/api/v5/neural/state':
                self._handle_neural_state(query)
            elif path == '/api/v5/neural/history':
                self._handle_neural_history(query)
            
            # Routes for pattern processing
            elif path == '/api/v5/patterns/latest':
                self._handle_latest_patterns(query)
            elif path == '/api/v5/patterns/metrics':
                self._handle_pattern_metrics(query)
            
            # Routes for consciousness analytics
            elif path == '/api/v5/consciousness/metrics':
                self._handle_consciousness_metrics(query)
            elif path == '/api/v5/consciousness/nodes':
                self._handle_consciousness_nodes(query)
            
            # Routes for language memory integration
            elif path == '/api/v5/language/topic':
                self._handle_language_topic(query)
            elif path == '/api/v5/language/stats':
                self._handle_language_stats(query)
            
            # System status and metadata
            elif path == '/api/v5/system/status':
                self._handle_system_status()
            elif path == '/api/v5/system/plugins':
                self._handle_system_plugins()
            
            # Default response for unknown routes
            else:
                self._send_error(404, "Endpoint not found")
                
        except Exception as e:
            logger.error(f"Error handling API request: {str(e)}")
            self._send_error(500, f"Internal server error: {str(e)}")
    
    def do_POST(self):
        """Handle POST requests"""
        if not self.plugin:
            self._send_error(500, "Server configuration error")
            return
        
        # Get request body
        content_length = int(self.headers.get('Content-Length', 0))
        request_body = self.rfile.read(content_length).decode('utf-8')
        
        try:
            # Parse JSON body
            body_data = {}
            if request_body:
                body_data = json.loads(request_body)
            
            # Parse URL
            parsed_url = urlparse(self.path)
            path = parsed_url.path
            
            # Routes for neural state
            if path == '/api/v5/neural/request_update':
                self._handle_neural_update_request(body_data)
            
            # Routes for pattern processing
            elif path == '/api/v5/patterns/process':
                self._handle_process_patterns(body_data)
            
            # Routes for consciousness analytics
            elif path == '/api/v5/consciousness/analyze':
                self._handle_analyze_consciousness(body_data)
            
            # Routes for language memory integration
            elif path == '/api/v5/language/request_topic':
                self._handle_request_language_topic(body_data)
            
            # Default response for unknown routes
            else:
                self._send_error(404, "Endpoint not found")
                
        except json.JSONDecodeError:
            self._send_error(400, "Invalid JSON in request body")
        except Exception as e:
            logger.error(f"Error handling API request: {str(e)}")
            self._send_error(500, f"Internal server error: {str(e)}")
    
    def _send_response(self, data, status_code=200):
        """Send a JSON response"""
        self._set_headers(status_code)
        response = json.dumps(data).encode('utf-8')
        self.wfile.write(response)
    
    def _send_error(self, status_code, message):
        """Send an error response"""
        self._set_headers(status_code)
        error_data = {
            "error": True,
            "message": message,
            "status": status_code
        }
        response = json.dumps(error_data).encode('utf-8')
        self.wfile.write(response)
    
    def _handle_neural_state(self, query):
        """Handle request for latest neural state"""
        self.plugin.get_neural_state(callback=self._async_response_callback)
    
    def _handle_neural_history(self, query):
        """Handle request for neural state history"""
        count = int(query.get('count', [10])[0])
        self.plugin.get_neural_history(count, callback=self._async_response_callback)
    
    def _handle_neural_update_request(self, data):
        """Handle request to update neural state"""
        include_layers = data.get('include_layers', True)
        include_weights = data.get('include_weights', False)
        
        self.plugin.request_neural_update(
            include_layers, 
            include_weights, 
            callback=self._async_response_callback
        )
    
    def _handle_latest_patterns(self, query):
        """Handle request for latest patterns"""
        self.plugin.get_latest_patterns(callback=self._async_response_callback)
    
    def _handle_pattern_metrics(self, query):
        """Handle request for pattern metrics"""
        metric_types = query.get('types', [])
        if isinstance(metric_types, list) and len(metric_types) == 1:
            metric_types = metric_types[0].split(',')
        
        self.plugin.get_pattern_metrics(metric_types, callback=self._async_response_callback)
    
    def _handle_process_patterns(self, data):
        """Handle request to process patterns"""
        patterns = data.get('patterns', [])
        options = data.get('options', {})
        
        self.plugin.process_patterns(patterns, options, callback=self._async_response_callback)
    
    def _handle_consciousness_metrics(self, query):
        """Handle request for consciousness metrics"""
        metrics = query.get('metrics', [])
        if isinstance(metrics, list) and len(metrics) == 1:
            metrics = metrics[0].split(',')
        
        history = query.get('history', ['false'])[0].lower() == 'true'
        count = int(query.get('count', [10])[0])
        
        self.plugin.get_consciousness_metrics(
            metrics, 
            history, 
            count, 
            callback=self._async_response_callback
        )
    
    def _handle_consciousness_nodes(self, query):
        """Handle request for consciousness node data"""
        self.plugin.get_consciousness_nodes(callback=self._async_response_callback)
    
    def _handle_analyze_consciousness(self, data):
        """Handle request to analyze consciousness"""
        neural_data = data.get('neural_data', {})
        options = data.get('options', {})
        
        self.plugin.analyze_consciousness(
            neural_data, 
            options, 
            callback=self._async_response_callback
        )
    
    def _handle_language_topic(self, query):
        """Handle request for language topic data"""
        topic = query.get('topic', ['default'])[0]
        self.plugin.get_language_topic(topic, callback=self._async_response_callback)
    
    def _handle_language_stats(self, query):
        """Handle request for language memory statistics"""
        self.plugin.get_language_stats(callback=self._async_response_callback)
    
    def _handle_request_language_topic(self, data):
        """Handle request to process a language topic"""
        topic = data.get('topic', 'default')
        options = data.get('options', {})
        
        self.plugin.request_language_topic(
            topic, 
            options, 
            callback=self._async_response_callback
        )
    
    def _handle_system_status(self):
        """Handle request for system status"""
        self.plugin.get_system_status(callback=self._async_response_callback)
    
    def _handle_system_plugins(self):
        """Handle request for system plugins"""
        self.plugin.get_system_plugins(callback=self._async_response_callback)
    
    def _async_response_callback(self, response_data):
        """Callback for asynchronous responses"""
        if isinstance(response_data, dict) and response_data.get('error'):
            self._send_error(
                response_data.get('status', 500),
                response_data.get('message', 'Internal error')
            )
        else:
            self._send_response(response_data)


class APIServicePlugin:
    """Socket-ready plugin for API services"""
    
    def __init__(self, node_id="api_service", host="0.0.0.0", port=5000):
        """
        Initialize the API service plugin
        
        Args:
            node_id: Unique identifier for this plugin
            host: Host address to bind the server to
            port: Port number to listen on
        """
        self.node_id = node_id
        self.socket = NodeSocket(node_id, "service")
        
        # Server configuration
        self.host = host
        self.port = port
        self.server = None
        self.server_thread = None
        
        # Request tracking
        self.pending_requests = {}
        self.request_timeout = 15  # seconds
        
        # Register with discovery service if available
        if HAS_DISCOVERY:
            self.client = register_node(self)
            logger.info(f"Registered plugin with discovery service: {node_id}")
        else:
            self.client = None
            logger.warning(f"Running without discovery service: {node_id}")
        
        # Set up socket message handlers
        self.socket.register_message_handler("neural_state_response", self._handle_neural_state_response)
        self.socket.register_message_handler("pattern_response", self._handle_pattern_response)
        self.socket.register_message_handler("consciousness_analysis_response", self._handle_consciousness_response)
        self.socket.register_message_handler("metrics_response", self._handle_metrics_response)
        self.socket.register_message_handler("language_topic_response", self._handle_language_response)
        self.socket.register_message_handler("system_status_response", self._handle_system_status_response)
        
        # Plugin metadata
        self.plugin_type = "v5_plugin"
        self.plugin_subtype = "api_service"
        self.api_version = "v5.0"
        
        # Start the HTTP server
        self._start_server()
        
        logger.info(f"API Service Plugin initialized: {node_id}, listening on {host}:{port}")
    
    def _start_server(self):
        """Start the HTTP server"""
        try:
            # Create a request handler class with a reference to this plugin
            def handler(*args, **kwargs):
                return APIRequestHandler(*args, plugin=self, **kwargs)
            
            # Create and start the server
            self.server = ThreadingHTTPServer((self.host, self.port), handler)
            self.server_thread = threading.Thread(target=self.server.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            logger.info(f"API server started on {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Failed to start API server: {str(e)}")
            self.server = None
    
    def stop(self):
        """Stop the API service"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            logger.info("API server stopped")
    
    def _generate_request_id(self):
        """Generate a unique request ID"""
        return str(uuid.uuid4())
    
    def _register_pending_request(self, callback):
        """
        Register a pending request with a callback
        
        Args:
            callback: Function to call with the response
            
        Returns:
            Request ID
        """
        request_id = self._generate_request_id()
        self.pending_requests[request_id] = {
            "callback": callback,
            "timestamp": time.time()
        }
        return request_id
    
    def _handle_response(self, message, request_id_field="request_id", data_field="data", error_field="error"):
        """
        Handle a response message for a pending request
        
        Args:
            message: Response message
            request_id_field: Field name for the request ID
            data_field: Field name for the response data
            error_field: Field name for error information
        """
        request_id = message.get(request_id_field)
        if not request_id:
            logger.warning(f"Received response without request ID: {message.get('type')}")
            return
        
        pending_request = self.pending_requests.pop(request_id, None)
        if not pending_request:
            logger.warning(f"Received response for unknown request: {request_id}")
            return
        
        callback = pending_request.get("callback")
        if not callback:
            logger.warning(f"No callback for request: {request_id}")
            return
        
        # Check for error
        if error_field in message:
            error_data = {
                "error": True,
                "message": message.get(error_field),
                "status": 500
            }
            callback(error_data)
            return
        
        # Extract data
        response_data = message.get(data_field, {})
        callback(response_data)
    
    def _handle_neural_state_response(self, message):
        """Handle neural state response"""
        self._handle_response(message)
    
    def _handle_pattern_response(self, message):
        """Handle pattern processing response"""
        self._handle_response(message)
    
    def _handle_consciousness_response(self, message):
        """Handle consciousness analysis response"""
        self._handle_response(message)
    
    def _handle_metrics_response(self, message):
        """Handle metrics response"""
        self._handle_response(message)
    
    def _handle_language_response(self, message):
        """Handle language memory response"""
        self._handle_response(message)
    
    def _handle_system_status_response(self, message):
        """Handle system status response"""
        self._handle_response(message)
    
    def get_neural_state(self, callback):
        """
        Get the latest neural state
        
        Args:
            callback: Function to call with the response
        """
        request_id = self._register_pending_request(callback)
        
        self.socket.send_message({
            "type": "get_neural_state",
            "request_id": request_id,
            "content": {
                "latest": True
            }
        })
    
    def get_neural_history(self, count, callback):
        """
        Get historical neural state data
        
        Args:
            count: Number of historical states to retrieve
            callback: Function to call with the response
        """
        request_id = self._register_pending_request(callback)
        
        self.socket.send_message({
            "type": "get_neural_state",
            "request_id": request_id,
            "content": {
                "latest": False,
                "count": count
            }
        })
    
    def request_neural_update(self, include_layers, include_weights, callback):
        """
        Request an update to the neural state
        
        Args:
            include_layers: Whether to include layer information
            include_weights: Whether to include weight information
            callback: Function to call with the response
        """
        request_id = self._register_pending_request(callback)
        
        self.socket.send_message({
            "type": "request_state_update",
            "request_id": request_id,
            "content": {
                "include_layers": include_layers,
                "include_weights": include_weights
            }
        })
    
    def get_latest_patterns(self, callback):
        """
        Get the latest patterns
        
        Args:
            callback: Function to call with the response
        """
        request_id = self._register_pending_request(callback)
        
        self.socket.send_message({
            "type": "get_patterns",
            "request_id": request_id,
            "content": {}
        })
    
    def get_pattern_metrics(self, metric_types, callback):
        """
        Get pattern metrics
        
        Args:
            metric_types: List of metric types to retrieve
            callback: Function to call with the response
        """
        request_id = self._register_pending_request(callback)
        
        self.socket.send_message({
            "type": "get_pattern_metrics",
            "request_id": request_id,
            "content": {
                "metric_types": metric_types
            }
        })
    
    def process_patterns(self, patterns, options, callback):
        """
        Process patterns
        
        Args:
            patterns: Pattern data to process
            options: Processing options
            callback: Function to call with the response
        """
        request_id = self._register_pending_request(callback)
        
        self.socket.send_message({
            "type": "process_patterns",
            "request_id": request_id,
            "content": {
                "patterns": patterns,
                "options": options
            }
        })
    
    def get_consciousness_metrics(self, metrics, history, count, callback):
        """
        Get consciousness metrics
        
        Args:
            metrics: List of specific metrics to retrieve
            history: Whether to include historical data
            count: Number of historical records to retrieve
            callback: Function to call with the response
        """
        request_id = self._register_pending_request(callback)
        
        self.socket.send_message({
            "type": "get_metrics",
            "request_id": request_id,
            "content": {
                "metrics": metrics,
                "history": history,
                "count": count
            }
        })
    
    def get_consciousness_nodes(self, callback):
        """
        Get consciousness node data
        
        Args:
            callback: Function to call with the response
        """
        request_id = self._register_pending_request(callback)
        
        self.socket.send_message({
            "type": "get_metrics",
            "request_id": request_id,
            "content": {
                "include_nodes": True
            }
        })
    
    def analyze_consciousness(self, neural_data, options, callback):
        """
        Analyze consciousness based on neural data
        
        Args:
            neural_data: Neural network state data
            options: Analysis options
            callback: Function to call with the response
        """
        request_id = self._register_pending_request(callback)
        
        self.socket.send_message({
            "type": "request_analysis",
            "request_id": request_id,
            "content": {
                "neural_data": neural_data,
                "options": options
            }
        })
    
    def get_language_topic(self, topic, callback):
        """
        Get language memory data for a specific topic
        
        Args:
            topic: Topic to retrieve
            callback: Function to call with the response
        """
        request_id = self._register_pending_request(callback)
        
        self.socket.send_message({
            "type": "get_topic",
            "request_id": request_id,
            "content": {
                "topic": topic
            }
        })
    
    def get_language_stats(self, callback):
        """
        Get language memory statistics
        
        Args:
            callback: Function to call with the response
        """
        request_id = self._register_pending_request(callback)
        
        self.socket.send_message({
            "type": "get_stats",
            "request_id": request_id,
            "content": {}
        })
    
    def request_language_topic(self, topic, options, callback):
        """
        Request processing of a language topic
        
        Args:
            topic: Topic to process
            options: Processing options
            callback: Function to call with the response
        """
        request_id = self._register_pending_request(callback)
        
        self.socket.send_message({
            "type": "process_topic",
            "request_id": request_id,
            "content": {
                "topic": topic,
                "options": options
            }
        })
    
    def get_system_status(self, callback):
        """
        Get system status information
        
        Args:
            callback: Function to call with the response
        """
        # For system status, we gather information locally instead of making a socket request
        system_status = {
            "status": "online",
            "version": "5.0.0",
            "api_version": self.api_version,
            "uptime": time.time() - self.start_time if hasattr(self, 'start_time') else 0,
            "plugins_connected": True,  # Assuming we're connected if the API is running
            "endpoints": [
                {"path": "/api/v5/neural/state", "method": "GET", "description": "Get latest neural state"},
                {"path": "/api/v5/neural/history", "method": "GET", "description": "Get neural state history"},
                {"path": "/api/v5/neural/request_update", "method": "POST", "description": "Request neural state update"},
                {"path": "/api/v5/patterns/latest", "method": "GET", "description": "Get latest patterns"},
                {"path": "/api/v5/patterns/metrics", "method": "GET", "description": "Get pattern metrics"},
                {"path": "/api/v5/patterns/process", "method": "POST", "description": "Process patterns"},
                {"path": "/api/v5/consciousness/metrics", "method": "GET", "description": "Get consciousness metrics"},
                {"path": "/api/v5/consciousness/nodes", "method": "GET", "description": "Get consciousness node data"},
                {"path": "/api/v5/consciousness/analyze", "method": "POST", "description": "Analyze consciousness"},
                {"path": "/api/v5/language/topic", "method": "GET", "description": "Get language topic data"},
                {"path": "/api/v5/language/stats", "method": "GET", "description": "Get language memory statistics"},
                {"path": "/api/v5/language/request_topic", "method": "POST", "description": "Request language topic processing"},
                {"path": "/api/v5/system/status", "method": "GET", "description": "Get system status"},
                {"path": "/api/v5/system/plugins", "method": "GET", "description": "Get system plugins"}
            ]
        }
        
        callback(system_status)
    
    def get_system_plugins(self, callback):
        """
        Get information about available plugins
        
        Args:
            callback: Function to call with the response
        """
        request_id = self._register_pending_request(callback)
        
        self.socket.send_message({
            "type": "get_connected_plugins",
            "request_id": request_id,
            "content": {}
        })
        
        # If we don't receive a response within a short time,
        # fall back to a local implementation
        threading.Timer(1.0, self._fallback_system_plugins, args=[request_id]).start()
    
    def _fallback_system_plugins(self, request_id):
        """
        Fallback handler for system plugins if no response is received
        
        Args:
            request_id: Original request ID
        """
        if request_id in self.pending_requests:
            # The request is still pending, so provide a fallback response
            pending_request = self.pending_requests.pop(request_id, None)
            if pending_request and pending_request.get("callback"):
                # Create minimal plugin information
                plugin_info = {
                    "plugins": [
                        {
                            "id": "neural_state_provider",
                            "type": "v5_plugin",
                            "subtype": "state_provider",
                            "status": "unknown"
                        },
                        {
                            "id": "pattern_processor",
                            "type": "v5_plugin",
                            "subtype": "processor",
                            "status": "unknown"
                        },
                        {
                            "id": "consciousness_analytics",
                            "type": "v5_plugin",
                            "subtype": "analytics",
                            "status": "unknown"
                        },
                        {
                            "id": "language_memory_integration",
                            "type": "v5_plugin",
                            "subtype": "integration",
                            "status": "unknown"
                        },
                        {
                            "id": self.node_id,
                            "type": self.plugin_type,
                            "subtype": self.plugin_subtype,
                            "status": "online",
                            "api_version": self.api_version
                        }
                    ]
                }
                
                pending_request["callback"](plugin_info)
    
    def cleanup_pending_requests(self):
        """Clean up expired pending requests"""
        current_time = time.time()
        expired_requests = []
        
        for request_id, request_info in self.pending_requests.items():
            if current_time - request_info.get("timestamp", 0) > self.request_timeout:
                expired_requests.append(request_id)
        
        for request_id in expired_requests:
            pending_request = self.pending_requests.pop(request_id)
            if pending_request and pending_request.get("callback"):
                # Send timeout error to callback
                error_data = {
                    "error": True,
                    "message": "Request timed out",
                    "status": 504
                }
                pending_request["callback"](error_data)
        
        # Schedule next cleanup
        if self.pending_requests:
            threading.Timer(self.request_timeout, self.cleanup_pending_requests).start()
    
    def get_socket_descriptor(self):
        """
        Return socket descriptor for frontend integration
        
        Returns:
            Socket descriptor dictionary
        """
        return {
            "plugin_id": self.node_id,
            "message_types": ["neural_state_response", "pattern_response", 
                             "consciousness_analysis_response", "metrics_response", 
                             "language_topic_response", "system_status_response"],
            "data_format": "json",
            "api_endpoints": f"http://localhost:{self.port}/api/v5"
        } 