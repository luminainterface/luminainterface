"""
Infection Module for Version 1
This module provides functionality to infect files and integrate with the node system.
"""

import os
import ast
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import importlib.util
import inspect
from ..nodes.node_implementation import HybridNode, CentralNode

logger = logging.getLogger(__name__)

class FileInfector:
    def __init__(self, target_directory: str):
        """
        Initialize the file infector.
        
        Args:
            target_directory: Directory to scan for files to infect
        """
        self.target_directory = Path(target_directory)
        self.infected_files: Dict[str, List[str]] = {}
        self.node_integration = None
        
        logger.info(f"Initialized FileInfector for directory: {target_directory}")
    
    def scan_directory(self) -> List[str]:
        """
        Scan the target directory for Python files.
        
        Returns:
            List of Python file paths
        """
        python_files = []
        for file_path in self.target_directory.rglob("*.py"):
            if file_path.is_file():
                python_files.append(str(file_path))
        
        logger.info(f"Found {len(python_files)} Python files in {self.target_directory}")
        return python_files
    
    def infect_file(self, file_path: str, infection_code: str) -> bool:
        """
        Infect a Python file with the given code.
        
        Args:
            file_path: Path to the file to infect
            infection_code: Code to inject into the file
            
        Returns:
            True if infection was successful, False otherwise
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the file to find appropriate injection points
            tree = ast.parse(content)
            
            # Find the last import statement
            last_import = None
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    last_import = node
            
            # Create new import statement
            new_import = ast.parse(infection_code)
            
            # Insert after the last import
            if last_import:
                tree.body.insert(tree.body.index(last_import) + 1, new_import.body[0])
            else:
                tree.body.insert(0, new_import.body[0])
            
            # Write back to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(ast.unparse(tree))
            
            if file_path not in self.infected_files:
                self.infected_files[file_path] = []
            self.infected_files[file_path].append(infection_code)
            
            logger.info(f"Successfully infected file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to infect file {file_path}: {str(e)}")
            return False
    
    def generate_infection_code(self, node_id: str) -> str:
        """
        Generate infection code that integrates with a specific node.
        
        Args:
            node_id: ID of the node to integrate with
            
        Returns:
            Infection code as a string
        """
        return f"""
from src.v1.nodes.node_implementation import HybridNode, CentralNode
from src.v1.infection.infection_module import FileInfector

# Initialize node integration
node_integration = FileInfector.get_node_integration()
if node_integration:
    node = node_integration.get_node('{node_id}')
    if node:
        # Add this file's functionality to the node
        node.add_module_functionality(__file__)
"""

class NodeIntegration:
    def __init__(self, central_node: CentralNode):
        """
        Initialize the node integration system.
        
        Args:
            central_node: Central node to integrate with
        """
        self.central_node = central_node
        self.module_functionality: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Initialized NodeIntegration")
    
    def register_module(self, file_path: str, functionality: Dict[str, Any]) -> None:
        """
        Register a module's functionality with the node system.
        
        Args:
            file_path: Path to the module file
            functionality: Dictionary of functionality to register
        """
        self.module_functionality[file_path] = functionality
        logger.info(f"Registered functionality from {file_path}")
    
    def get_node(self, node_id: str) -> Optional[HybridNode]:
        """
        Get a hybrid node by ID.
        
        Args:
            node_id: ID of the node to get
            
        Returns:
            HybridNode instance if found, None otherwise
        """
        return self.central_node.hybrid_nodes.get(node_id)
    
    def distribute_functionality(self) -> None:
        """
        Distribute registered functionality to all nodes.
        """
        for node_id, node in self.central_node.hybrid_nodes.items():
            for file_path, functionality in self.module_functionality.items():
                node.add_module_functionality(file_path, functionality)
        
        logger.info("Distributed functionality to all nodes")

class InfectedModule:
    def __init__(self, file_path: str, node_integration: NodeIntegration):
        """
        Initialize an infected module.
        
        Args:
            file_path: Path to the infected file
            node_integration: Node integration system
        """
        self.file_path = file_path
        self.node_integration = node_integration
        self.functionality = self._extract_functionality()
        
        logger.info(f"Initialized InfectedModule for {file_path}")
    
    def _extract_functionality(self) -> Dict[str, Any]:
        """
        Extract functionality from the infected module.
        
        Returns:
            Dictionary of extracted functionality
        """
        try:
            spec = importlib.util.spec_from_file_location("infected_module", self.file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            functionality = {
                'functions': {},
                'classes': {},
                'variables': {}
            }
            
            for name, obj in inspect.getmembers(module):
                if inspect.isfunction(obj):
                    functionality['functions'][name] = obj
                elif inspect.isclass(obj):
                    functionality['classes'][name] = obj
                elif not name.startswith('_'):
                    functionality['variables'][name] = obj
            
            return functionality
            
        except Exception as e:
            logger.error(f"Failed to extract functionality from {self.file_path}: {str(e)}")
            return {}
    
    def integrate_with_node(self, node_id: str) -> bool:
        """
        Integrate the module's functionality with a specific node.
        
        Args:
            node_id: ID of the node to integrate with
            
        Returns:
            True if integration was successful, False otherwise
        """
        node = self.node_integration.get_node(node_id)
        if node:
            self.node_integration.register_module(self.file_path, self.functionality)
            return True
        return False

# Global node integration instance
_node_integration = None

def get_node_integration() -> Optional[NodeIntegration]:
    """
    Get the global node integration instance.
    
    Returns:
        NodeIntegration instance if initialized, None otherwise
    """
    return _node_integration

def initialize_node_integration(central_node: CentralNode) -> None:
    """
    Initialize the global node integration system.
    
    Args:
        central_node: Central node to integrate with
    """
    global _node_integration
    _node_integration = NodeIntegration(central_node)
    logger.info("Initialized global node integration system") 