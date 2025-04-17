import sys
import os
import logging
from typing import Dict, Any
from central_node import CentralNode

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('backend_diagnostics.log')
        ]
    )
    return logging.getLogger('BackendDiagnostics')

def run_diagnostics() -> Dict[str, Any]:
    """Run comprehensive diagnostics on the backend"""
    logger = setup_logging()
    results = {
        'central_node': False,
        'nodes': {},
        'processors': {},
        'connections': False,
        'errors': []
    }
    
    try:
        # Test CentralNode initialization
        logger.info("Testing CentralNode initialization...")
        central_node = CentralNode()
        results['central_node'] = True
        
        # Test node initialization and activation
        logger.info("Testing nodes...")
        for node_name, node in central_node.nodes.items():
            node_status = {
                'initialized': False,
                'activated': False,
                'errors': []
            }
            
            try:
                # Check initialization
                if hasattr(node, 'is_initialized'):
                    node_status['initialized'] = node.is_initialized()
                elif hasattr(node, '_initialized'):
                    node_status['initialized'] = node._initialized
                
                # Check activation
                if hasattr(node, 'is_active'):
                    node_status['activated'] = node.is_active()
                elif hasattr(node, '_active'):
                    node_status['activated'] = node._active
                    
            except Exception as e:
                node_status['errors'].append(str(e))
                
            results['nodes'][node_name] = node_status
            
        # Test processor initialization and activation
        logger.info("Testing processors...")
        for proc_name, processor in central_node.processors.items():
            proc_status = {
                'initialized': False,
                'activated': False,
                'errors': []
            }
            
            try:
                # Check initialization
                if hasattr(processor, 'is_initialized'):
                    proc_status['initialized'] = processor.is_initialized()
                elif hasattr(processor, '_initialized'):
                    proc_status['initialized'] = processor._initialized
                
                # Check activation
                if hasattr(processor, 'is_active'):
                    proc_status['activated'] = processor.is_active()
                elif hasattr(processor, '_active'):
                    proc_status['activated'] = processor._active
                    
            except Exception as e:
                proc_status['errors'].append(str(e))
                
            results['processors'][proc_name] = proc_status
            
        # Test connections
        logger.info("Testing component connections...")
        try:
            connections = central_node.get_component_dependencies()
            results['connections'] = bool(connections)
        except Exception as e:
            results['errors'].append(f"Connection test failed: {str(e)}")
            
    except Exception as e:
        results['errors'].append(f"Critical error: {str(e)}")
        
    return results

def print_diagnostics_results(results: Dict[str, Any]):
    """Print diagnostic results in a readable format"""
    print("\n=== Backend Diagnostics Results ===\n")
    
    print("Central Node:", "✓" if results['central_node'] else "✗")
    
    print("\nNodes:")
    for node_name, status in results['nodes'].items():
        print(f"  {node_name}:")
        print(f"    Initialized: {'✓' if status['initialized'] else '✗'}")
        print(f"    Activated: {'✓' if status['activated'] else '✗'}")
        if status['errors']:
            print("    Errors:")
            for error in status['errors']:
                print(f"      - {error}")
                
    print("\nProcessors:")
    for proc_name, status in results['processors'].items():
        print(f"  {proc_name}:")
        print(f"    Initialized: {'✓' if status['initialized'] else '✗'}")
        print(f"    Activated: {'✓' if status['activated'] else '✗'}")
        if status['errors']:
            print("    Errors:")
            for error in status['errors']:
                print(f"      - {error}")
                
    print("\nConnections:", "✓" if results['connections'] else "✗")
    
    if results['errors']:
        print("\nCritical Errors:")
        for error in results['errors']:
            print(f"  - {error}")

if __name__ == "__main__":
    results = run_diagnostics()
    print_diagnostics_results(results) 
 
 