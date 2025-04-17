#!/usr/bin/env python3
"""
Central Language Node Launcher

This script launches the Central Language Node with appropriate configuration
for v5-v10 integration. It serves as the main entry point for running 
the unified language system.
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
import subprocess
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("launch_central_language_node.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("launch_central_language_node")

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import the Central Language Node
try:
    from src.central_language_node import CentralLanguageNode
    logger.info("Successfully imported CentralLanguageNode")
except ImportError as e:
    logger.error(f"Failed to import CentralLanguageNode: {str(e)}")
    logger.error("Please make sure central_language_node.py exists")
    sys.exit(1)


def create_default_config():
    """Create default configuration file if it doesn't exist"""
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    config_path = config_dir / "central_language_node.json"
    
    if config_path.exists():
        logger.info(f"Configuration file already exists at {config_path}")
        return str(config_path)
    
    default_config = {
        "data_path": "data/memory",
        "component_priorities": {
            "language_memory": 100,
            "conversation_memory": 90,
            "english_language_trainer": 80,
            "memory_synthesis": 70,
            "llm_enhancement": 60,
            "neural_network": 50,
            "memory_api": 40
        },
        "enable_v5_integration": True,
        "enable_v10_features": True,
        "consciousness_level": 2,  # 0-5 scale
        "memory_retention_days": 365,
        "auto_synthesis_interval_hours": 24,
        "v10_integration": {
            "register_with_central_node": True,
            "provide_consciousness_data": True,
            "consciousness_contribution_weight": 0.8
        },
        "api_server": {
            "enabled": True,
            "host": "0.0.0.0",
            "port": 8000,
            "debug": False
        }
    }
    
    # Write the config file
    with open(config_path, 'w') as f:
        json.dump(default_config, f, indent=2)
    
    logger.info(f"Created default configuration at {config_path}")
    return str(config_path)


def check_dependencies():
    """Check for required dependencies"""
    logger.info("Checking dependencies...")
    
    missing_dependencies = []
    optional_dependencies = []
    
    # Check for required components
    required_components = [
        {"module": "src.language_memory_synthesis_integration", "name": "Language Memory Synthesis"},
        {"module": "src.conversation_memory", "name": "Conversation Memory"},
        {"module": "src.language.language_memory", "name": "Language Memory"}
    ]
    
    for component in required_components:
        try:
            __import__(component["module"])
            logger.info(f"✅ Found required component: {component['name']}")
        except ImportError:
            logger.error(f"❌ Missing required component: {component['name']}")
            missing_dependencies.append(component["name"])
    
    # Check for optional v5/v10 components
    optional_components = [
        {"module": "src.v5.language_memory_integration", "name": "V5 Language Integration"},
        {"module": "src.v5_integration.visualization_bridge", "name": "V5 Visualization Bridge"},
        {"module": "src.consciousness_node", "name": "V10 Consciousness Node"}
    ]
    
    for component in optional_components:
        try:
            __import__(component["module"])
            logger.info(f"✅ Found optional component: {component['name']}")
        except ImportError:
            logger.warning(f"⚠️ Missing optional component: {component['name']}")
            optional_dependencies.append(component["name"])
    
    return missing_dependencies, optional_dependencies


def start_api_server(node, config):
    """Start the memory API server in a separate process if enabled"""
    api_config = config.get("api_server", {})
    
    if not api_config.get("enabled", False):
        logger.info("API server is disabled in configuration")
        return None
    
    # Check if memory_api_server component is available
    if node.get_component("memory_api_server") is None:
        logger.error("Cannot start API server: memory_api_server component not available")
        return None
    
    logger.info("Starting memory API server in a separate process...")
    
    try:
        # Prepare command to run the server
        host = api_config.get("host", "0.0.0.0")
        port = api_config.get("port", 8000)
        debug = "--debug" if api_config.get("debug", False) else ""
        
        cmd = [sys.executable, "-m", "src.memory_api_server", "--host", host, "--port", str(port)]
        if debug:
            cmd.append(debug)
        
        # Start the server process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Give it a moment to start
        time.sleep(1)
        
        # Check if process is still running
        if process.poll() is None:
            logger.info(f"Memory API server started on {host}:{port}")
            return process
        else:
            stdout, stderr = process.communicate()
            logger.error(f"Failed to start Memory API server: {stderr}")
            return None
    except Exception as e:
        logger.error(f"Error starting Memory API server: {str(e)}")
        return None


def start_visualization_server(node, config):
    """Start the V5 visualization server if available"""
    if not config.get("enable_v5_integration", True):
        logger.info("V5 integration is disabled in configuration")
        return None
    
    # Check if V5 visualization is available
    v5_bridge = node.get_component("v5_bridge")
    if v5_bridge is None:
        logger.info("V5 visualization bridge not available")
        return None
    
    logger.info("Starting V5 visualization server...")
    
    try:
        # Check if the fractal_analytics.py script exists
        fractal_script_path = Path("src/v5/fractal_analytics.py")
        if not fractal_script_path.exists():
            logger.warning(f"V5 fractal analytics script not found at {fractal_script_path}")
            return None
        
        # Start the visualization server process
        process = subprocess.Popen(
            [sys.executable, str(fractal_script_path), "--server"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Give it a moment to start
        time.sleep(1)
        
        # Check if process is still running
        if process.poll() is None:
            logger.info("V5 visualization server started")
            return process
        else:
            stdout, stderr = process.communicate()
            logger.error(f"Failed to start V5 visualization server: {stderr}")
            return None
    except Exception as e:
        logger.error(f"Error starting V5 visualization server: {str(e)}")
        return None


def main():
    """Main function for launching the Central Language Node"""
    parser = argparse.ArgumentParser(description="Launch the Central Language Node")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--skip-checks", action="store_true", help="Skip dependency checks")
    parser.add_argument("--disable-api", action="store_true", help="Disable API server")
    parser.add_argument("--disable-v5", action="store_true", help="Disable V5 integration")
    parser.add_argument("--disable-v10", action="store_true", help="Disable V10 integration")
    parser.add_argument("--test", action="store_true", help="Run test operations and exit")
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("CENTRAL LANGUAGE NODE LAUNCHER")
    print("="*80 + "\n")
    
    # Create default config if needed
    config_path = args.config or create_default_config()
    
    # Check dependencies unless skipped
    if not args.skip_checks:
        missing, optional = check_dependencies()
        if missing:
            print("\n❌ Missing required dependencies. Cannot continue.")
            sys.exit(1)
    
    # Load configuration
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {str(e)}")
        print(f"\n❌ Failed to load configuration: {str(e)}")
        sys.exit(1)
    
    # Apply command-line overrides
    if args.disable_api:
        config["api_server"]["enabled"] = False
    
    if args.disable_v5:
        config["enable_v5_integration"] = False
    
    if args.disable_v10:
        config["enable_v10_features"] = False
    
    # Create the Central Language Node
    try:
        node = CentralLanguageNode(config_path=config_path)
        status = node.get_status()
        
        print(f"\n✅ Central Language Node initialized with {len(status['active_components'])}/{len(status['component_status'])} components")
        
        # Print active components
        print("\nActive Components:")
        for component in status['active_components']:
            print(f"  - {component}")
        
        # Print inactive components
        inactive = [name for name, status in status['component_status'].items() 
                   if status != "active"]
        if inactive:
            print("\nInactive Components:")
            for component in inactive:
                print(f"  - {component}")
    except Exception as e:
        logger.error(f"Error creating Central Language Node: {str(e)}")
        print(f"\n❌ Failed to initialize Central Language Node: {str(e)}")
        sys.exit(1)
    
    # Start API server if enabled
    api_process = None
    if config["api_server"]["enabled"]:
        api_process = start_api_server(node, config)
    
    # Start V5 visualization server if enabled
    v5_process = None
    if config["enable_v5_integration"]:
        v5_process = start_visualization_server(node, config)
    
    # Run test operations if requested
    if args.test:
        # Import the test functionality
        from src.central_language_node import main as run_tests
        run_tests()
        
        # Clean up processes
        if api_process:
            api_process.terminate()
        if v5_process:
            v5_process.terminate()
        
        sys.exit(0)
    
    print("\n" + "="*80)
    print("CENTRAL LANGUAGE NODE RUNNING")
    print("="*80)
    print("\nPress Ctrl+C to stop...\n")
    
    # Keep the script running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        
        # Clean up processes
        if api_process:
            api_process.terminate()
        if v5_process:
            v5_process.terminate()
        
        print("Goodbye!")


if __name__ == "__main__":
    main() 