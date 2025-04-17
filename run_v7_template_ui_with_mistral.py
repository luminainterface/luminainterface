#!/usr/bin/env python3
"""
Run the V7 Template UI with Mistral and Consciousness System Plugins

This script launches the V7 Template UI and loads both the Mistral Neural Chat 
and Consciousness System plugins for an integrated neural network experience.
"""

import os
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("V7TemplateRunner")

def main():
    """Run the V7 Template UI with Mistral Neural Chat and Consciousness plugins"""
    logger.info("Starting V7 Template UI with integrated plugins")
    
    # Ensure we can import from the current directory
    script_dir = Path(__file__).parent.absolute()
    sys.path.insert(0, str(script_dir))
    
    # Create necessary data directories
    data_dirs = ["data", "data/consciousness", "data/mistral", "data/neural"]
    for dir_path in data_dirs:
        Path(dir_path).mkdir(exist_ok=True)
        logger.info(f"Ensured directory exists: {dir_path}")
    
    # Set environment variables for configuration
    os.environ["TEMPLATE_PLUGINS_ENABLED"] = "true"
    os.environ["TEMPLATE_PLUGINS_DIRS"] = "plugins;src/v7/plugins;src/plugins"
    os.environ["TEMPLATE_AUTO_LOAD_PLUGINS"] = "consciousness_system_plugin.py;mistral_plugin.py;neural_network_plugin.py"
    os.environ["TEMPLATE_TITLE"] = "LUMINA V7 Integrated Consciousness System"
    os.environ["TEMPLATE_ICON"] = "icons/neural_icon.png"
    
    logger.info("Environment variables configured for plugin loading")
    
    # Try to import the template launcher
    try:
        from src.v7.template.launcher import launch_template_ui
        
        # Launch the template UI
        launch_template_ui(
            plugins_dir="plugins",
            load_plugins=["consciousness_system_plugin.py", "mistral_plugin.py", "neural_network_plugin.py"],
            theme="dark"
        )
        
    except ImportError as e:
        logger.error(f"Failed to import the template launcher: {e}")
        logger.info("Trying alternative launch method...")
        
        # Try running the batch file if available
        batch_file = Path("run_v7_template_ui_with_plugins.bat")
        if os.name == "nt" and batch_file.exists():
            logger.info(f"Running batch file: {batch_file}")
            os.system(str(batch_file))
        elif os.name == "nt" and Path("run_v7_template_ui.bat").exists():
            # Fall back to original batch file
            logger.info("Running original template batch file")
            os.system("run_v7_template_ui.bat")
        else:
            # Try direct module import
            try:
                from v7_pyside6_template import main as run_template
                
                # Run the template directly
                logger.info("Running template directly via main function")
                return run_template()
                
            except ImportError as e2:
                logger.error(f"Failed to run template UI: {e2}")
                print("ERROR: Could not launch the V7 Template UI.")
                print("Please ensure the V7 template is properly installed.")
                print("You can try running the batch file manually:")
                print("    .\\run_v7_template_ui_with_plugins.bat")
                return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 