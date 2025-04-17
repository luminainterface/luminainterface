#!/usr/bin/env python3
'''
Database Connection Verification

This script verifies and manages database connections across the system.
'''

# Import required modules
import os
import sys
import time
import logging
import argparse
import importlib.util
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

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

def check_language_database_bridge():
    '''Check if the Language Database Bridge module is available'''
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
    '''Check if the Database Connection Manager module is available'''
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
    '''Register the Language Database Bridge with the Connection Manager'''
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

def check_bridge_sync_status() -> Tuple[bool, Dict[str, Any]]:
    '''
    Check the synchronization status of the Language Database Bridge
    
    Returns:
        Tuple of (operational_status, status_dict)
    '''
    try:
        from language.language_database_bridge import get_language_database_bridge
        bridge = get_language_database_bridge()
        
        # Get status
        status = bridge.get_status()
        
        # Check if it's operational
        operational = (
            status.get("running", False) and
            isinstance(status.get("sync_stats"), dict) and
            status.get("sync_stats", {}).get("total_syncs", 0) > 0
        )
        
        return operational, status
    except Exception as e:
        logger.error(f"Error checking bridge sync status: {e}")
        return False, {"error": str(e)}

def parse_arguments():
    '''Parse command line arguments'''
    parser = argparse.ArgumentParser(description="Verify database connections")
    
    parser.add_argument('--sync-now', action='store_true',
                        help='Force immediate synchronization')
    
    parser.add_argument('--optimize', action='store_true',
                        help='Optimize database connections')
    
    parser.add_argument('--components', type=str, default=None,
                        help='Comma-separated list of components to verify')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed status information')
                        
    parser.add_argument('--interval', type=int, default=None,
                        help='Set synchronization interval in seconds')
    
    return parser.parse_args()

def main():
    '''Main entry point'''
    # Parse command line arguments
    args = parse_arguments()
    
    # Start verification
    logger.info("Starting database connection verification")
    start_time = time.time()
    
    # Track discovered components
    discovered_components = {}
    
    # Check for Language Database Bridge
    bridge_available, bridge_status = check_language_database_bridge()
    discovered_components["language_database_bridge"] = bridge_available
    
    if not bridge_available:
        logger.warning(f"Language Database Bridge module is not available: {bridge_status}")
    else:
        logger.info(f"Language Database Bridge module is available: {bridge_status}")
    
    # Check for Database Connection Manager
    manager_available, manager_status = check_database_connection_manager()
    discovered_components["database_connection_manager"] = manager_available
    
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
                
            # Check bridge synchronization status
            sync_operational, sync_status = check_bridge_sync_status()
            if sync_operational:
                logger.info("Bridge synchronization is operational")
                if args.verbose:
                    logger.info(f"Bridge status: {sync_status}")
            else:
                logger.warning("Bridge synchronization is not operational")
                if args.verbose:
                    logger.warning(f"Bridge status details: {sync_status}")
                    
            # Set synchronization interval if specified
            if args.interval is not None and args.interval >= 60:
                try:
                    from language.language_database_bridge import get_language_database_bridge
                    bridge = get_language_database_bridge()
                    bridge.set_sync_interval(args.interval)
                    logger.info(f"Set synchronization interval to {args.interval} seconds")
                except Exception as e:
                    logger.error(f"Failed to set synchronization interval: {e}")
        
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
        
        # Optimize connections if requested
        if args.optimize:
            logger.info("Optimizing database connections")
            # Implementation would go here
        
        # Show verification time
        elapsed = time.time() - start_time
        logger.info(f"Verification completed in {elapsed:.2f} seconds")
        
        # Exit with success
        return 0
        
    except ImportError:
        logger.warning("Database connection manager not found in language module")
        
    except Exception as e:
        logger.error(f"Error loading database connection manager: {e}")
    
    logger.error("Failed to load database connection manager")
    return 1

if __name__ == "__main__":
    sys.exit(main())

                logger.warning(f"Failed to register Language Database Bridge: {reg_status}")
                
            # Check bridge synchronization status
            sync_operational, sync_status = check_bridge_sync_status()
            if sync_operational:
                logger.info("Bridge synchronization is operational")
                if args.verbose:
                    logger.info(f"Bridge status: {sync_status}")
            else:
                logger.warning("Bridge synchronization is not operational")
                if args.verbose:
                    logger.warning(f"Bridge status details: {sync_status}")
                    
            # Set synchronization interval if specified
            if args.interval is not None and args.interval >= 60:
                try:
                    from language.language_database_bridge import get_language_database_bridge
                    bridge = get_language_database_bridge()
                    bridge.set_sync_interval(args.interval)
                    logger.info(f"Set synchronization interval to {args.interval} seconds")
                except Exception as e:
                    logger.error(f"Failed to set synchronization interval: {e}")
        
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
        
        # Optimize connections if requested
        if args.optimize:
            logger.info("Optimizing database connections")
            # Implementation would go here
        
        # Show verification time
        elapsed = time.time() - start_time
        logger.info(f"Verification completed in {elapsed:.2f} seconds")
        
        # Exit with success
        return 0
        
    except ImportError:
        logger.warning("Database connection manager not found in language module")
        
    except Exception as e:
        logger.error(f"Error loading database connection manager: {e}")
    
    logger.error("Failed to load database connection manager")
    return 1

if __name__ == "__main__":
    sys.exit(main())
