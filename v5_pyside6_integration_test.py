#!/usr/bin/env python3
"""
Test Script for V5 Fractal Echo Visualization + Language Memory System Integration with PySide6

This script performs a comprehensive set of tests to verify the proper integration
of the V5 Visualization system with the Language Memory System using PySide6.
"""

import os
import sys
import time
import logging
import argparse
import traceback
import json
from datetime import datetime
from pathlib import Path

# Configure logging
log_file = "v5_language_memory_test.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("V5-LM-Test")

# Make sure we're in a valid environment
try:
    from PySide6.QtCore import Signal, Slot, QObject, QTimer, Qt
    from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
    from PySide6.QtGui import QColor
except ImportError:
    logger.critical("PySide6 is not installed. Please install it with: pip install PySide6")
    sys.exit(1)

# Try to import our bridge module and other components
try:
    from language_memory_v5_bridge import LanguageMemoryV5Bridge, get_bridge
except ImportError:
    logger.critical("language_memory_v5_bridge module not found. Please make sure it's available.")
    sys.exit(1)

# Test results tracking
class TestResults:
    """Class to track and report test results"""
    
    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.test_logs = []
        
    def log_test(self, test_name, success, message=""):
        """Log a test result"""
        self.total_tests += 1
        result = "PASS" if success else "FAIL"
        
        if success:
            self.passed_tests += 1
            logger.info(f"✅ {test_name}: {result} - {message}")
        else:
            self.failed_tests += 1
            logger.error(f"❌ {test_name}: {result} - {message}")
            
        self.test_logs.append({
            "name": test_name,
            "result": result,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
    
    def summary(self):
        """Get a summary of test results"""
        return {
            "total": self.total_tests,
            "passed": self.passed_tests,
            "failed": self.failed_tests,
            "success_rate": (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        }
    
    def print_summary(self):
        """Print a summary of test results"""
        summary = self.summary()
        logger.info(f"Test Summary: {summary['passed']}/{summary['total']} tests passed ({summary['success_rate']:.1f}%)")
        if summary['failed'] > 0:
            logger.warning(f"Failed tests: {summary['failed']}")
            for test in self.test_logs:
                if test['result'] == 'FAIL':
                    logger.warning(f"  - {test['name']}: {test['message']}")

# Test functions
def test_qt_compatibility(results):
    """Test Qt compatibility and signal/slot mechanism"""
    logger.info("Testing Qt compatibility")
    
    try:
        # Create a QApplication instance for testing
        if QApplication.instance() is None:
            test_app = QApplication([])
        else:
            test_app = QApplication.instance()
        
        # Create a test signal/slot connection
        class TestObject(QObject):
            test_signal = Signal(str)
            
            def __init__(self):
                super().__init__()
                self.received_message = None
                
            @Slot(str)
            def test_slot(self, message):
                self.received_message = message
        
        # Create test objects and connect
        sender = TestObject()
        receiver = TestObject()
        
        sender.test_signal.connect(receiver.test_slot)
        
        # Emit a test signal
        test_message = "Test message via Qt signal"
        sender.test_signal.emit(test_message)
        
        # Verify reception
        success = receiver.received_message == test_message
        
        results.log_test(
            "Qt compatibility check",
            success,
            f"Signal/slot mechanism working correctly: {receiver.received_message}" if success else "Signal was not received correctly"
        )
        
        return success
    except Exception as e:
        results.log_test(
            "Qt compatibility check",
            False,
            f"Exception during Qt test: {str(e)}"
        )
        logger.error(f"Exception during Qt test: {traceback.format_exc()}")
        return False

def test_socket_system(results):
    """Test the V5 socket system with NodeSocket"""
    logger.info("Testing socket system")
    
    try:
        # Import the socket provider
        from language_memory_v5_bridge import MemoryAPISocketProvider
        
        # Create socket instances
        socket_a = MemoryAPISocketProvider(mock_mode=True)
        socket_b = MemoryAPISocketProvider(mock_mode=True)
        
        # Customize node IDs
        socket_a.node_id = "test_socket_a"
        socket_b.node_id = "test_socket_b"
        
        # Connect them
        connect_success = socket_a.connect_to(socket_b)
        
        # Test message receipt
        message_received = False
        
        def test_handler(message):
            nonlocal message_received
            if message.get("type") == "test_message" and message.get("data", {}).get("content") == "Hello from socket A":
                message_received = True
        
        # Register handler
        socket_b.register_message_handler("test_message", test_handler)
        
        # Start processing
        socket_b.start_processing()
        
        # Send a test message
        send_success = socket_a.send_message(
            "test_message", 
            {"content": "Hello from socket A"},
            target_id=socket_b.node_id
        )
        
        # Wait a bit for message processing
        time.sleep(0.5)
        
        # Stop processing
        socket_b.stop_processing()
        
        # Check results
        connection_success = connect_success and send_success
        
        results.log_test(
            "Socket connection",
            connection_success,
            "Successfully created and connected sockets" if connection_success else "Failed to connect sockets"
        )
        
        results.log_test(
            "Socket message handler",
            message_received,
            "Successfully received and processed message" if message_received else "Failed to receive or process message"
        )
        
        return connection_success and message_received
    except Exception as e:
        results.log_test(
            "Socket system test",
            False,
            f"Exception during socket test: {str(e)}"
        )
        logger.error(f"Exception during socket test: {traceback.format_exc()}")
        return False

def test_language_memory_bridge(results):
    """Test the Language Memory V5 Bridge initialization and methods"""
    logger.info("Testing Language Memory V5 Bridge")
    
    try:
        # Create a bridge instance in mock mode
        bridge = LanguageMemoryV5Bridge(mock_mode=True)
        
        # Test connection
        connection_success = bridge.connect()
        
        # Check for required methods
        required_methods = [
            "get_topics",
            "search",
            "get_memory",
            "store_memory",
            "get_connections",
            "generate_fractal"
        ]
        
        missing_methods = []
        for method in required_methods:
            if not hasattr(bridge, method) or not callable(getattr(bridge, method)):
                missing_methods.append(method)
        
        methods_success = len(missing_methods) == 0
        
        # Test basic functionality
        functionality_success = False
        test_query = "neural networks"
        
        try:
            topics = bridge.get_topics()
            search_results = bridge.search(test_query, limit=2)
            
            # Store a test memory
            memory_id = bridge.store_memory(
                content="Test memory for V5 integration testing",
                topic="visualization_techniques",
                tags=["test", "v5", "integration"]
            )
            
            # Retrieve the memory
            memory = bridge.get_memory(memory_id)
            
            # Get connections
            connections = bridge.get_connections(memory_id)
            
            # Generate a fractal
            fractal = bridge.generate_fractal(memory_id)
            
            functionality_success = (
                isinstance(topics, list) and 
                isinstance(search_results, list) and
                isinstance(memory, dict) and
                isinstance(connections, list) and
                isinstance(fractal, dict) and
                len(fractal.get("nodes", [])) > 0
            )
        except Exception as e:
            logger.error(f"Error testing bridge functionality: {e}")
            functionality_success = False
        
        # Disconnect
        bridge.disconnect()
        
        # Log results
        results.log_test(
            "Bridge connection",
            connection_success,
            "Successfully connected to Language Memory API (mock)" if connection_success 
            else "Failed to connect to Language Memory API"
        )
        
        results.log_test(
            "Bridge required methods",
            methods_success,
            "All required methods present" if methods_success 
            else f"Missing required methods: {', '.join(missing_methods)}"
        )
        
        results.log_test(
            "Bridge basic functionality",
            functionality_success,
            "Successfully performed basic operations" if functionality_success 
            else "Failed to perform basic operations"
        )
        
        return connection_success and methods_success and functionality_success
    except Exception as e:
        results.log_test(
            "Language Memory Bridge test",
            False,
            f"Exception during bridge test: {str(e)}"
        )
        logger.error(f"Exception during bridge test: {traceback.format_exc()}")
        return False

def test_memory_api_compat(results):
    """Test Memory API compatibility"""
    logger.info("Testing Memory API compatibility")
    
    try:
        # Get the singleton bridge instance
        bridge = get_bridge(mock_mode=True)
        
        # Test that get_bridge returns the same instance
        bridge2 = get_bridge()
        singleton_success = bridge is bridge2
        
        # Test API functions
        api_functions = [
            ("get_topics", []),
            ("search", ["test query"]),
            ("store_memory", ["Test content"])
        ]
        
        api_success = True
        for func_name, args in api_functions:
            try:
                func = getattr(bridge, func_name)
                result = func(*args)
                # Just check that it doesn't crash
                logger.debug(f"API function {func_name} returned: {result}")
            except Exception as e:
                logger.error(f"API function {func_name} failed: {e}")
                api_success = False
                break
        
        results.log_test(
            "Memory API singleton",
            singleton_success,
            "get_bridge() returns the same instance as expected" if singleton_success 
            else "get_bridge() returns different instances"
        )
        
        results.log_test(
            "Memory API functions",
            api_success,
            "All API functions executed without errors" if api_success 
            else f"API function execution failed"
        )
        
        return singleton_success and api_success
    except Exception as e:
        results.log_test(
            "Memory API compatibility test",
            False,
            f"Exception during API test: {str(e)}"
        )
        logger.error(f"Exception during API test: {traceback.format_exc()}")
        return False

def test_frontend_components(results):
    """Test frontend components integration"""
    logger.info("Testing frontend components")
    
    # Mock classes if they don't exist yet
    class MockFrontendSocketManager:
        def __init__(self):
            self.node_id = "frontend_socket_manager"
            self.sockets = {}
        
        def register_socket(self, socket):
            self.sockets[socket.node_id] = socket
            return True
        
        def receive_message(self, message):
            pass  # Just for interface compatibility
    
    class MockV5MainWidget(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.layout = QVBoxLayout(self)
            self.setLayout(self.layout)
            self.panels = []
        
        def add_panel(self, panel_widget):
            self.layout.addWidget(panel_widget)
            self.panels.append(panel_widget)
            return True
    
    try:
        # Try to import actual components if they exist
        try:
            from v5_frontend import FrontendSocketManager, V5MainWidget
        except ImportError:
            logger.warning("Actual frontend components not found, using mocks")
            FrontendSocketManager = MockFrontendSocketManager
            V5MainWidget = MockV5MainWidget
        
        # Test socket manager creation
        socket_manager = FrontendSocketManager()
        
        # Get bridge instance
        bridge = get_bridge(mock_mode=True)
        
        # Connect bridge to socket manager if compatible
        socket_connect_success = False
        try:
            if hasattr(bridge.socket, 'connect_to') and hasattr(socket_manager, 'receive_message'):
                socket_connect_success = bridge.socket.connect_to(socket_manager)
        except Exception as e:
            logger.error(f"Error connecting bridge to socket manager: {e}")
            socket_connect_success = False
        
        # Create an application if needed
        if QApplication.instance() is None:
            app = QApplication([])
        
        # Create main widget
        main_widget = V5MainWidget()
        
        # Test UI panel loading
        ui_panel_success = len(main_widget.panels) >= 0  # No panels is OK initially
        
        # Add a mock panel to test
        class MockPanel(QWidget):
            def __init__(self, parent=None):
                super().__init__(parent)
        
        panel_add_success = main_widget.add_panel(MockPanel())
        
        results.log_test(
            "Frontend socket manager",
            socket_manager is not None,
            "Successfully created frontend socket manager" if socket_manager is not None 
            else "Failed to create frontend socket manager"
        )
        
        results.log_test(
            "Bridge-frontend connection",
            socket_connect_success,
            "Successfully connected bridge to frontend" if socket_connect_success 
            else "Failed to connect bridge to frontend"
        )
        
        results.log_test(
            "V5 main widget",
            main_widget is not None,
            "Successfully created V5 main widget" if main_widget is not None 
            else "Failed to create V5 main widget"
        )
        
        results.log_test(
            "UI panel system",
            panel_add_success,
            "Successfully added UI panel" if panel_add_success 
            else "Failed to add UI panel"
        )
        
        return (socket_manager is not None and 
                main_widget is not None and 
                panel_add_success)
    except Exception as e:
        results.log_test(
            "Frontend components test",
            False,
            f"Exception during frontend test: {str(e)}"
        )
        logger.error(f"Exception during frontend test: {traceback.format_exc()}")
        return False

def try_full_system_initialization(results):
    """Try to initialize the full system as an integrated test"""
    logger.info("Testing full system initialization")
    
    try:
        # Create QApplication if necessary
        if QApplication.instance() is None:
            app = QApplication([])
        else:
            app = QApplication.instance()
        
        # Create main window
        main_window = QMainWindow()
        main_window.setWindowTitle("V5 Language Memory Integration Test")
        main_window.resize(1024, 768)
        
        # Try to import actual V5MainWidget or use mock
        try:
            from v5_frontend import V5MainWidget
        except ImportError:
            # Use mock class
            class V5MainWidget(QWidget):
                def __init__(self, parent=None):
                    super().__init__(parent)
                    self.layout = QVBoxLayout(self)
                    self.setLayout(self.layout)
                    
                    # Add a label to indicate this is a mock
                    from PySide6.QtWidgets import QLabel
                    label = QLabel("Mock V5 Visualization (actual component not found)")
                    label.setAlignment(Qt.AlignCenter)
                    self.layout.addWidget(label)
        
        # Create main widget
        main_widget = V5MainWidget()
        main_window.setCentralWidget(main_widget)
        
        # Initialize bridge
        bridge = get_bridge(mock_mode=True)
        connected = bridge.connect()
        
        # Short timer to close the window automatically
        QTimer.singleShot(2000, main_window.close)
        
        # Show the window (briefly)
        main_window.show()
        
        # Process events
        app.processEvents()
        
        # Success if we got this far
        full_init_success = connected and main_window.isVisible()
        
        results.log_test(
            "Full system initialization",
            full_init_success,
            "Successfully initialized full system" if full_init_success 
            else "Failed to initialize full system"
        )
        
        return full_init_success
    except Exception as e:
        results.log_test(
            "Full system initialization test",
            False,
            f"Exception during system initialization: {str(e)}"
        )
        logger.error(f"Exception during system initialization: {traceback.format_exc()}")
        return False

def run_all_tests():
    """Run all tests and return the results"""
    logger.info("Running all integration tests")
    results = TestResults()
    
    # Run each test
    qt_success = test_qt_compatibility(results)
    socket_success = test_socket_system(results)
    bridge_success = test_language_memory_bridge(results)
    api_success = test_memory_api_compat(results)
    frontend_success = test_frontend_components(results)
    full_success = try_full_system_initialization(results)
    
    # Log summary of results
    logger.info("All tests completed")
    results.print_summary()
    
    # Save test results to JSON file
    results_file = "v5_integration_test_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "summary": results.summary(),
            "tests": results.test_logs
        }, f, indent=2)
    
    logger.info(f"Test results saved to {results_file}")
    
    return results.summary()["failed"] == 0

def run_interactive_test():
    """Run the application in interactive mode for manual testing"""
    logger.info("Starting interactive test mode")
    
    try:
        # Create QApplication if necessary
        if QApplication.instance() is None:
            app = QApplication([])
        else:
            app = QApplication.instance()
        
        # Create main window
        main_window = QMainWindow()
        main_window.setWindowTitle("V5 Language Memory Integration - Interactive Test")
        main_window.resize(1200, 800)
        
        # Try to import actual V5MainWidget or use mock
        try:
            from v5_frontend import V5MainWidget
            logger.info("Using actual V5MainWidget")
        except ImportError:
            logger.warning("Using mock V5MainWidget")
            # Use mock class
            class V5MainWidget(QWidget):
                def __init__(self, parent=None):
                    super().__init__(parent)
                    self.layout = QVBoxLayout(self)
                    self.setLayout(self.layout)
                    
                    # Add a label to indicate this is a mock
                    from PySide6.QtWidgets import QLabel, QPushButton
                    label = QLabel("Mock V5 Visualization (actual component not found)")
                    label.setAlignment(Qt.AlignCenter)
                    self.layout.addWidget(label)
                    
                    # Add a button to test the bridge
                    button = QPushButton("Test Bridge Connection")
                    button.clicked.connect(self.test_bridge)
                    self.layout.addWidget(button)
                
                def test_bridge(self):
                    try:
                        bridge = get_bridge(mock_mode=True)
                        connected = bridge.connect()
                        topics = bridge.get_topics()
                        
                        from PySide6.QtWidgets import QMessageBox
                        QMessageBox.information(
                            self,
                            "Bridge Test",
                            f"Connected: {connected}\nTopics: {', '.join(topics)}"
                        )
                    except Exception as e:
                        from PySide6.QtWidgets import QMessageBox
                        QMessageBox.critical(
                            self,
                            "Bridge Test Error",
                            f"Error: {str(e)}"
                        )
        
        # Create main widget
        main_widget = V5MainWidget()
        main_window.setCentralWidget(main_widget)
        
        # Initialize bridge
        bridge = get_bridge(mock_mode=True)
        connected = bridge.connect()
        logger.info(f"Bridge connected: {connected}")
        
        # Show the window
        main_window.show()
        
        # Start Qt event loop
        logger.info("Starting interactive test. Close the window to exit.")
        return app.exec()
    except Exception as e:
        logger.error(f"Error in interactive test: {str(e)}")
        logger.error(traceback.format_exc())
        return 1

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="V5 Language Memory Integration Test")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.interactive:
        sys.exit(run_interactive_test())
    else:
        success = run_all_tests()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 