#!/usr/bin/env python
"""
V5 Enhanced Launcher

Enhanced launcher for V5/V6/V7 integration that connects the V5 Fractal Echo,
V6 Portal of Contradiction, and V7 Node Consciousness systems.
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("V5Enhanced")

# Import Qt compatibility layer
try:
    from src.v5.ui.qt_compat import QtWidgets, QtCore, QtGui, Qt
    # Ensure QTimer is imported
    QTimer = QtCore.QTimer
except ImportError:
    logger.warning("V5 Qt compatibility layer not found. Using direct PySide6 imports.")
    try:
        from PySide6 import QtWidgets, QtCore, QtGui
        from PySide6.QtCore import Qt
        # Ensure QTimer is imported
        QTimer = QtCore.QTimer
    except ImportError:
        logger.error("PySide6 not found. Please install PySide6 or configure the V5 Qt compatibility layer.")
        sys.exit(1)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="V5/V6/V7 Enhanced Integration Launcher")
    
    parser.add_argument('--fix-message-flow', action='store_true',
                        help='Enable message flow fixes')
    
    parser.add_argument('--pattern', type=str, default='auto',
                        choices=['auto', 'manual', 'guided'],
                        help='Pattern generation mode')
    
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with additional logging')
    
    # Add V6-specific arguments
    parser.add_argument('--enable-breath', action='store_true',
                        help='Enable breath integration')
    
    parser.add_argument('--enable-glyphs', action='store_true',
                        help='Enable glyph field overlay')
    
    parser.add_argument('--enable-mirror', action='store_true',
                        help='Enable mirror mode for contradiction handling')
    
    # Add V7 specific arguments
    parser.add_argument('--enable-v7', action='store_true',
                        help='Enable V7 Node Consciousness features')
    
    parser.add_argument('--monday', action='store_true',
                        help='Enable Monday consciousness integration')
    
    parser.add_argument('--auto-wiki', action='store_true',
                        help='Enable AutoWiki learning system')
    
    # Add mock mode argument
    parser.add_argument('--mock', action='store_true',
                        help='Enable mock mode for development')
    
    return parser.parse_args()

def configure_logging(debug_mode):
    """Configure logging based on debug mode"""
    if debug_mode:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    else:
        logging.getLogger().setLevel(logging.INFO)

class V5V6Bridge:
    """Bridge between V5 and V6 systems"""
    def __init__(self, fix_message_flow=False, pattern_mode='auto', debug=False):
        self.fix_message_flow = fix_message_flow
        self.pattern_mode = pattern_mode
        self.debug = debug
        self.pattern_cache = {}
        self.message_queue = []
        self.is_processing = False
        self.socket_manager = None
        
        # Timer for simulating V5 backend processes
        self.process_timer = QTimer()
        self.process_timer.timeout.connect(self.process_patterns)
        self.process_timer.start(2000)  # Process every 2 seconds
        
        logger.info(f"V5V6Bridge initialized: fix_flow={fix_message_flow}, pattern={pattern_mode}")
    
    def process_patterns(self):
        """Process pattern recognition (simulated)"""
        if self.is_processing:
            return
            
        self.is_processing = True
        
        # Simulate pattern processing
        if self.pattern_mode == 'auto' and len(self.message_queue) > 0:
            message = self.message_queue.pop(0)
            logger.debug(f"Processing pattern for: {message[:30]}...")
            
            # Simulate pattern recognition
            pattern_data = {
                'pattern_id': f"p{int(time.time())}",
                'strength': round(0.5 + (hash(message) % 100) / 200, 2),
                'type': ['symbolic', 'emotional', 'resonant'][hash(message) % 3],
                'nodes': [
                    {'id': 'n1', 'activation': round(0.3 + (hash(message[:5]) % 100) / 200, 2)},
                    {'id': 'n2', 'activation': round(0.4 + (hash(message[5:10]) % 100) / 200, 2)},
                    {'id': 'n3', 'activation': round(0.2 + (hash(message[10:15]) % 100) / 200, 2)}
                ]
            }
            
            # Store in cache
            self.pattern_cache[pattern_data['pattern_id']] = pattern_data
            
            # Emit pattern recognized event
            if self.socket_manager and self.socket_manager.is_connected():
                self.socket_manager.emit('pattern_recognized', pattern_data)
                logger.debug(f"Emitted pattern: {pattern_data['pattern_id']}")
        
        self.is_processing = False
    
    def set_socket_manager(self, socket_manager):
        """Set the socket manager for communication"""
        self.socket_manager = socket_manager
        
        # Register handlers
        if socket_manager and socket_manager.is_connected():
            socket_manager.register_handler('message_received', self.handle_message_received)
            socket_manager.register_handler('request_patterns', self.handle_request_patterns)
            logger.debug("Registered bridge handlers with socket manager")
    
    def handle_message_received(self, data):
        """Handle received messages"""
        if 'message' in data:
            message = data['message']
            logger.debug(f"Received message: {message[:30]}...")
            
            # Apply message flow fix if enabled
            if self.fix_message_flow:
                # Normalize message flow
                message = self.normalize_message(message)
            
            # Add to queue for pattern processing
            self.message_queue.append(message)
    
    def handle_request_patterns(self, data):
        """Handle request for pattern data"""
        logger.debug("Received request for patterns")
        
        if self.socket_manager and self.socket_manager.is_connected():
            self.socket_manager.emit('patterns_data', {
                'patterns': list(self.pattern_cache.values())
            })
    
    def normalize_message(self, message):
        """Apply message flow fixes"""
        # This would contain actual message flow fixes in the real implementation
        return message

def initialize_v6_components(config):
    """Initialize V6 components for the Portal of Contradiction"""
    try:
        # Import V6 socket manager
        from src.v6.socket_manager import V6SocketManager, create_mock_plugins
        
        # Create socket manager
        socket_manager = V6SocketManager(mock_mode=config.get("mock_mode", True))
        
        # Import version bridge manager
        from src.v6.version_bridge_manager import VersionBridgeManager
        
        # Create bridge manager
        bridge_manager = VersionBridgeManager(config)
        
        # Import symbolic state manager
        from src.v6.symbolic_state_manager import V6SymbolicStateManager
        
        # Create symbolic state manager
        symbolic_state_manager = V6SymbolicStateManager(socket_manager)
        
        # Start breath cycle if enabled
        if config.get("enable_breath"):
            symbolic_state_manager.start_breath_cycle()
        
        # Load mock plugins if in mock mode
        if config.get("mock_mode"):
            plugins = create_mock_plugins()
            for plugin in plugins:
                socket_manager.register_plugin(plugin)
        
        # Initialize WebSocket connections
        if config.get("enable_v7"):
            socket_manager.connect_websocket("ws://localhost:8765/v7/consciousness", "consciousness_panel")
        
        socket_manager.connect_websocket("ws://localhost:8765/v6/glyphs", "glyph_panel")
        socket_manager.connect_websocket("ws://localhost:8765/v6/breath", "breath_panel")
        socket_manager.connect_websocket("ws://localhost:8765/v6/mirror", "mirror_panel")
        socket_manager.connect_websocket("ws://localhost:8765/v6/echo", "echo_panel")
        socket_manager.connect_websocket("ws://localhost:8765/v6/mythos", "mythos_panel")
        socket_manager.connect_websocket("ws://localhost:8765/v6/embodiment", "embodiment_panel")
        
        # Start bridge manager
        bridge_manager.start()
        
        logger.info("✅ V6 Portal of Contradiction components initialized")
        
        return {
            "socket_manager": socket_manager,
            "bridge_manager": bridge_manager,
            "symbolic_state_manager": symbolic_state_manager
        }
        
    except ImportError as e:
        logger.error(f"Error importing V6 components: {e}")
        return None
    except Exception as e:
        logger.error(f"Error initializing V6 components: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

def initialize_v6v7_connector(enable_v7=False, enable_monday=False, enable_auto_wiki=False, debug=False):
    """Initialize the V6-V7 connector if V7 is enabled"""
    if not enable_v7:
        logger.info("V7 features disabled, skipping V6-V7 connector initialization")
        return None
    
    try:
        # Import the V6-V7 connector
        from src.v7.v6_v7_connector import V6V7Connector
        
        # Configure the connector
        config = {
            "mock_mode": False,
            "v6_enabled": True,
            "v7_enabled": True,
            "debug": debug,
            "monday_integration_enabled": enable_monday,
            "auto_wiki_enabled": enable_auto_wiki
        }
        
        # Create and initialize the connector
        connector = V6V7Connector(config)
        success = connector.initialize()
        
        if success:
            logger.info("✅ V6-V7 Connector initialized successfully")
            return connector
        else:
            logger.warning("❌ V6-V7 Connector initialization failed")
            return None
            
    except ImportError:
        logger.warning("❌ V6-V7 Connector module not found")
        return None
    except Exception as e:
        logger.error(f"Error initializing V6-V7 Connector: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

def determine_ui_widget(enable_v7=False):
    """Determine which UI widget class to use based on configuration"""
    use_v6 = False
    use_v7 = enable_v7
    
    # Check if V7 is available and requested
    if use_v7:
        try:
            from src.v7.v7_integration import V7MainWidget
            logger.info("✅ Using V7 Main Widget")
            return V7MainWidget, False
        except ImportError:
            logger.warning("⚠️ V7 integration not available, falling back to V6")
            use_v7 = False
            use_v6 = True
    
    # Check if V6 is available
    if not use_v7:
        try:
            from src.v6.ui.main_widget import V6MainWidget
            logger.info("✅ Using V6 Main Widget")
            return V6MainWidget, True
        except ImportError:
            logger.warning("⚠️ V6 UI not available, falling back to V5")
            use_v6 = False
    
    # Fall back to V5
    from src.v5.ui.main_widget import V5MainWidget
    logger.info("ℹ️ Using V5 Main Widget")
    return V5MainWidget, False

def main():
    """Main entry point for the application"""
    # Parse arguments
    args = parse_arguments()
    
    # Configure logging
    configure_logging(args.debug)
    
    # Create config dictionary
    config = {
        "fix_message_flow": args.fix_message_flow,
        "pattern_mode": args.pattern,
        "debug": args.debug,
        "enable_breath": args.enable_breath,
        "enable_glyphs": args.enable_glyphs,
        "enable_mirror": args.enable_mirror,
        "enable_v7": args.enable_v7,
        "enable_monday": args.monday,
        "enable_auto_wiki": args.auto_wiki,
        "mock_mode": args.mock
    }
    
    # Log startup information
    logger.info(f"Starting V5 Enhanced Integration")
    logger.info(f"Arguments: fix_message_flow={args.fix_message_flow}, pattern={args.pattern}, debug={args.debug}")
    logger.info(f"V6 features: breath={args.enable_breath}, glyphs={args.enable_glyphs}, mirror={args.enable_mirror}")
    logger.info(f"V7 features: enabled={args.enable_v7}, monday={args.monday}, auto_wiki={args.auto_wiki}")
    
    # Create Qt application
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Lumina Neural Network")
    app.setStyle("Fusion")  # Use Fusion style for better cross-platform appearance
    
    # Set up dark palette for the application
    palette = app.palette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(22, 33, 51))
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(255, 255, 255))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(16, 26, 40))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(26, 38, 52))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(255, 255, 255))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor(255, 255, 255))
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(255, 255, 255))
    palette.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
    palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(0, 0, 0))
    app.setPalette(palette)
    
    # Initialize V6 components
    v6_components = initialize_v6_components(config)
    
    if v6_components:
        socket_manager = v6_components["socket_manager"]
        bridge_manager = v6_components["bridge_manager"]
        symbolic_state_manager = v6_components["symbolic_state_manager"]
    else:
        # Fall back to simple socket manager
        from src.v6.socket_manager import V6SocketManager
        socket_manager = V6SocketManager(mock_mode=True)
        bridge_manager = None
        symbolic_state_manager = None
        logger.warning("Using fallback socket manager")
    
    # Create V5/V6 bridge
    bridge = V5V6Bridge(
        fix_message_flow=args.fix_message_flow,
        pattern_mode=args.pattern,
        debug=args.debug
    )
    bridge.set_socket_manager(socket_manager)
    
    # Initialize V6-V7 connector if V7 is enabled
    v6v7_connector = None
    if args.enable_v7:
        v6v7_connector = initialize_v6v7_connector(
            enable_v7=args.enable_v7, 
            enable_monday=args.monday, 
            enable_auto_wiki=args.auto_wiki,
            debug=args.debug
        )
    
    # Determine which UI widget to use
    MainWidgetClass, use_v6 = determine_ui_widget(enable_v7=args.enable_v7)
    
    try:
        # Create main widget
        try:
            if args.enable_v7:
                # V7-style initialization - needs socket_manager, symbolic_state_manager, and v6v7_connector
                try:
                    main_widget = MainWidgetClass(
                        socket_manager=socket_manager,
                        symbolic_state_manager=symbolic_state_manager,
                        v6v7_connector=v6v7_connector,
                        config=config
                    )
                except TypeError:
                    logger.error("❌ V7-style initialization failed")
                    raise
            else:
                # Try V6-style initialization
                try:
                    main_widget = MainWidgetClass(socket_manager)
                except Exception as e:
                    logger.error(f"Error initializing V6 widget: {e}")
                    raise
        except Exception as e:
            logger.error(f"Error initializing main widget: {e}")
            raise
        
        # Configure window
        window_title = "Lumina Neural Network"
        if args.enable_v7:
            window_title += " - V7 Node Consciousness"
        else:
            window_title += " - V6 Portal of Contradiction"
            
        main_widget.setWindowTitle(window_title)
        main_widget.resize(1500, 950)
        main_widget.show()
        
        # Start V6-V7 connector if available
        if v6v7_connector:
            v6v7_connector.start()
        
        # Set up language memory bridge if available
        language_memory_bridge = None
        if bridge_manager:
            language_memory_bridge = bridge_manager.get_component("language_memory_v5_bridge")
            if language_memory_bridge:
                language_memory_bridge.set_socket_manager(socket_manager)
        
        # Simulate initial data to populate panels
        QTimer.singleShot(500, lambda: socket_manager.emit('system_ready', {
            'version': 'v6.0.2',
            'mode': args.pattern,
            'fix_flow': args.fix_message_flow,
            'timestamp': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }))
        
        # Run the application
        sys.exit(app.exec())
        
    except ImportError as e:
        logger.error(f"Error importing UI components: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error starting application: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        # Clean up
        if bridge_manager:
            bridge_manager.stop()

if __name__ == "__main__":
    main() 