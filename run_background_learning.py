#!/usr/bin/env python3
"""
Run Background Learning Engine

This script starts the background learning engine in standalone mode,
allowing it to run continuously and improve the system over time.
"""

import os
import sys
import time
import json
import logging
import argparse
import signal
from typing import Dict, Any
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/logs/background_learning.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BackgroundLearningRunner")

# Create required directories
os.makedirs("data/logs", exist_ok=True)

# Add project root to path if needed
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the background learning engine
try:
    from language.background_learning_engine import BackgroundLearningEngine, get_background_learning_engine
except ImportError:
    logger.error("Could not import BackgroundLearningEngine. Check your installation.")
    sys.exit(1)

# Global variables
learning_engine = None
running = True

def handle_signal(sig, frame):
    """Signal handler for graceful shutdown"""
    global running
    logger.info(f"Received signal {sig}, shutting down...")
    running = False

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run the background learning engine")
    
    parser.add_argument('--config', '-c', type=str, default=None,
                       help='Path to configuration JSON file')
    
    parser.add_argument('--interval', '-i', type=int, default=15,
                       help='Learning interval in minutes (default: 15)')
    
    parser.add_argument('--autowiki', '-a', action='store_true',
                       help='Enable autowiki learning (default: false)')
    
    parser.add_argument('--daemon', '-d', action='store_true',
                       help='Run in daemon mode (background)')
    
    parser.add_argument('--status', '-s', action='store_true',
                       help='Print status and exit')
    
    parser.add_argument('--clear-history', type=int, default=None, metavar='DAYS',
                       help='Clear learning history older than specified days')
    
    return parser.parse_args()

def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from file"""
    config = {}
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    return config

def update_config_from_args(config: Dict[str, Any], args) -> Dict[str, Any]:
    """Update configuration with command line arguments"""
    if not config.get("learning"):
        config["learning"] = {}
    
    # Update interval if specified
    if args.interval:
        config["learning"]["learning_interval_minutes"] = args.interval
    
    # Update autowiki setting if specified
    if args.autowiki:
        config["learning"]["enable_autowiki_learning"] = True
    
    return config

def create_pid_file():
    """Create PID file for daemon mode"""
    pid = os.getpid()
    with open("data/background_learning.pid", "w") as f:
        f.write(str(pid))
    logger.info(f"Created PID file with process ID {pid}")

def remove_pid_file():
    """Remove PID file"""
    if os.path.exists("data/background_learning.pid"):
        os.remove("data/background_learning.pid")
        logger.info("Removed PID file")

def print_status():
    """Print current status of the learning engine"""
    # Initialize engine if needed
    global learning_engine
    if not learning_engine:
        learning_engine = get_background_learning_engine()
    
    # Get status
    status = learning_engine.get_status()
    stats = learning_engine.get_statistics()
    
    # Print status
    print("\n=== Background Learning Engine Status ===\n")
    print(f"Running: {status['running']}")
    print(f"Initialized: {status['initialized']}")
    print(f"Learning Rate: {status['learning_rate']:.4f}")
    print(f"Queue Size: {status['queue_size']}")
    print(f"Learning Cycles: {stats['learning_cycles']}")
    print(f"Patterns Extracted: {stats['patterns_extracted']}")
    print(f"Concepts Learned: {stats['concepts_learned']}")
    print(f"AutoWiki Entries: {stats['autowiki_entries']}")
    print(f"Neural Adaptations: {stats['neural_adaptations']}")
    
    if stats['last_cycle']:
        last_cycle_time = datetime.fromisoformat(stats['last_cycle'])
        now = datetime.now()
        time_diff = now - last_cycle_time
        print(f"Last Cycle: {time_diff.total_seconds() / 60:.1f} minutes ago")
        print(f"Average Cycle Time: {stats['avg_cycle_time_ms']:.2f} ms")
    
    print("\nConnected Components:")
    for component, status in status['components'].items():
        print(f"  {component}: {'Connected' if status else 'Not Connected'}")
    
    print("\n")

def clear_history(days: int):
    """Clear learning history"""
    # Initialize engine if needed
    global learning_engine
    if not learning_engine:
        learning_engine = get_background_learning_engine()
    
    # Clear history
    removed = learning_engine.clear_learning_history(days_to_keep=days)
    print(f"Cleared {removed} items from learning history (kept last {days} days)")

def main():
    """Main entry point"""
    global learning_engine, running
    
    # Parse arguments
    args = parse_arguments()
    
    # Handle status request
    if args.status:
        print_status()
        return
    
    # Handle history clearing
    if args.clear_history is not None:
        clear_history(args.clear_history)
        return
    
    # Load configuration
    config = load_config(args.config)
    config = update_config_from_args(config, args)
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    # Create PID file for daemon mode
    if args.daemon:
        create_pid_file()
    
    try:
        # Initialize the background learning engine
        logger.info("Initializing background learning engine")
        learning_engine = get_background_learning_engine(config)
        
        # Start the engine
        logger.info("Starting background learning engine")
        success = learning_engine.start()
        
        if not success:
            logger.error("Failed to start background learning engine")
            return
        
        logger.info("Background learning engine started successfully")
        
        # Main loop - keep alive and monitor
        while running:
            time.sleep(10)
            
            # Print status periodically in interactive mode
            if not args.daemon and running:
                status = learning_engine.get_status()
                stats = learning_engine.get_statistics()
                logger.info(f"Learning cycles: {stats['learning_cycles']}, " +
                           f"Queue size: {status['queue_size']}, " +
                           f"Patterns: {stats['patterns_extracted']}, " +
                           f"Concepts: {stats['concepts_learned']}")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
    
    finally:
        # Stop the engine
        if learning_engine:
            logger.info("Stopping background learning engine")
            learning_engine.stop()
        
        # Remove PID file
        if args.daemon:
            remove_pid_file()
        
        logger.info("Background learning engine stopped")

if __name__ == "__main__":
    main() 