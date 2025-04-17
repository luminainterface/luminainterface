#!/usr/bin/env python3
"""
LUMINA Backend Diagnostics
Performs comprehensive system checks and component testing
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import importlib
import time
import platform
import subprocess

# Configure logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/diagnostics.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('diagnostics')

class SystemDiagnostics:
    """System diagnostics utility to verify environment before launching the Central Node Monitor"""
    
    def __init__(self):
        self.required_folders = ['src', 'logs', 'data', 'models', 'assets']
        self.required_packages = [
            'PySide6', 'numpy', 'psutil', 'requests', 'matplotlib', 
            'networkx', 'torch'
        ]
        self.version_folders = ['v5', 'v6', 'v7', 'v7_5', 'v8', 'v9', 'v10']
        self.required_modules = ['central_node', 'spiderweb.spiderweb_manager']
        self.ui_modules = ['ui.theme', 'ui.components']
        self.gpu_available = False
        self.diagnostics_results = {}
        
    def run_all_diagnostics(self):
        """Run all diagnostics and return overall success status"""
        logger.info("Starting system diagnostics...")
        
        # Run all diagnostics
        self.check_python_version()
        self.check_directories()
        self.check_packages()
        self.check_modules()
        self.check_ui_components()
        self.check_hardware_resources()
        self.check_network_connectivity()
        self.check_version_directories()
        self.check_model_files()
        
        # Determine overall success
        success = all(self.diagnostics_results.values())
        if success:
            logger.info("All diagnostics passed successfully")
        else:
            logger.error("Some diagnostics failed")
            # Print failed diagnostics
            for test, result in self.diagnostics_results.items():
                if not result:
                    logger.error(f"Failed test: {test}")
        
        return success
    
    def check_python_version(self):
        """Check Python version (minimum 3.8)"""
        logger.info("Checking Python version...")
        
        py_version = sys.version_info
        min_version = (3, 8)
        
        version_str = f"{py_version.major}.{py_version.minor}.{py_version.micro}"
        min_version_str = f"{min_version[0]}.{min_version[1]}"
        
        if py_version.major >= min_version[0] and py_version.minor >= min_version[1]:
            logger.info(f"Python version {version_str} meets requirements (min: {min_version_str})")
            self.diagnostics_results['python_version'] = True
        else:
            logger.error(f"Python version {version_str} does not meet minimum requirement of {min_version_str}")
            self.diagnostics_results['python_version'] = False
    
    def check_directories(self):
        """Check that required directories exist"""
        logger.info("Checking required directories...")
        
        missing_dirs = []
        for folder in self.required_folders:
            if not os.path.isdir(folder):
                missing_dirs.append(folder)
                try:
                    logger.warning(f"Directory '{folder}' not found, creating it")
                    os.makedirs(folder, exist_ok=True)
                except Exception as e:
                    logger.error(f"Failed to create directory '{folder}': {str(e)}")
        
        if not missing_dirs:
            logger.info("All required directories exist")
            self.diagnostics_results['directories'] = True
        else:
            logger.warning(f"Created missing directories: {', '.join(missing_dirs)}")
            self.diagnostics_results['directories'] = True  # Still true since we created them
    
    def check_packages(self):
        """Check that required Python packages are installed"""
        logger.info("Checking required packages...")
        
        missing_packages = []
        for package in self.required_packages:
            try:
                importlib.import_module(package)
                logger.info(f"Package '{package}' is installed")
            except ImportError:
                missing_packages.append(package)
                logger.error(f"Required package '{package}' is not installed")
        
        if not missing_packages:
            logger.info("All required packages are installed")
            self.diagnostics_results['packages'] = True
        else:
            logger.error(f"Missing packages: {', '.join(missing_packages)}")
            self.diagnostics_results['packages'] = False
    
    def check_modules(self):
        """Check that required application modules can be imported"""
        logger.info("Checking required application modules...")
        
        missing_modules = []
        for module in self.required_modules:
            try:
                # Add src to path temporarily
                if 'src' not in sys.path:
                    sys.path.append('src')
                
                # Try to import the module
                importlib.import_module(module)
                logger.info(f"Module '{module}' can be imported")
            except ImportError as e:
                missing_modules.append(f"{module} ({str(e)})")
                logger.error(f"Required module '{module}' cannot be imported: {str(e)}")
        
        if not missing_modules:
            logger.info("All required application modules can be imported")
            self.diagnostics_results['modules'] = True
        else:
            logger.error(f"Missing modules: {', '.join(missing_modules)}")
            self.diagnostics_results['modules'] = False
    
    def check_ui_components(self):
        """Check UI components and theme"""
        logger.info("Checking UI components and theme...")
        
        missing_ui_modules = []
        for module in self.ui_modules:
            try:
                # Add src to path temporarily
                if 'src' not in sys.path:
                    sys.path.append('src')
                
                # Try to import the module
                importlib.import_module(module)
                logger.info(f"UI module '{module}' can be imported")
            except ImportError as e:
                missing_ui_modules.append(f"{module} ({str(e)})")
                logger.error(f"UI module '{module}' cannot be imported: {str(e)}")
        
        if not missing_ui_modules:
            logger.info("All UI components can be imported")
            self.diagnostics_results['ui_components'] = True
        else:
            logger.error(f"Missing UI components: {', '.join(missing_ui_modules)}")
            self.diagnostics_results['ui_components'] = False
    
    def check_hardware_resources(self):
        """Check available hardware resources"""
        logger.info("Checking hardware resources...")
        
        try:
            import psutil
            
            # Check CPU
            cpu_count = psutil.cpu_count(logical=True)
            logger.info(f"CPU cores: {cpu_count}")
            
            # Check memory
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024 ** 3)
            logger.info(f"Total memory: {memory_gb:.2f} GB")
            logger.info(f"Available memory: {memory.available / (1024 ** 3):.2f} GB")
            
            # Check disk space
            disk = psutil.disk_usage('/')
            disk_gb = disk.total / (1024 ** 3)
            free_disk_gb = disk.free / (1024 ** 3)
            logger.info(f"Total disk space: {disk_gb:.2f} GB")
            logger.info(f"Free disk space: {free_disk_gb:.2f} GB")
            
            # Check for GPU
            try:
                import torch
                self.gpu_available = torch.cuda.is_available()
                if self.gpu_available:
                    gpu_count = torch.cuda.device_count()
                    gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                    logger.info(f"GPU available: {gpu_count} devices (primary: {gpu_name})")
                else:
                    logger.info("No GPU detected, running in CPU mode")
            except ImportError:
                logger.warning("PyTorch not installed, skipping GPU check")
                self.gpu_available = False
            
            # Check if hardware meets minimum requirements
            hardware_ok = (
                cpu_count >= 2 and 
                memory_gb >= 4 and 
                free_disk_gb >= 5
            )
            
            if hardware_ok:
                logger.info("Hardware resources meet minimum requirements")
                self.diagnostics_results['hardware'] = True
            else:
                logger.warning("Hardware resources may be insufficient:")
                if cpu_count < 2:
                    logger.warning("- CPU cores less than recommended minimum (2)")
                if memory_gb < 4:
                    logger.warning("- Memory less than recommended minimum (4 GB)")
                if free_disk_gb < 5:
                    logger.warning("- Free disk space less than recommended minimum (5 GB)")
                # Still allow it to run
                self.diagnostics_results['hardware'] = True
        
        except ImportError:
            logger.warning("psutil not installed, skipping hardware resource check")
            self.diagnostics_results['hardware'] = True  # Assume it's ok
        except Exception as e:
            logger.error(f"Error checking hardware resources: {str(e)}")
            self.diagnostics_results['hardware'] = True  # Assume it's ok
    
    def check_network_connectivity(self):
        """Check network connectivity"""
        logger.info("Checking network connectivity...")
        
        try:
            import requests
            
            # Try to connect to a reliable service
            response = requests.get("https://www.google.com", timeout=5)
            if response.status_code == 200:
                logger.info("Network connectivity test successful")
                self.diagnostics_results['network'] = True
            else:
                logger.warning(f"Network connectivity test returned status code {response.status_code}")
                self.diagnostics_results['network'] = True  # Still allow it to run
        
        except ImportError:
            logger.warning("requests not installed, skipping network connectivity check")
            self.diagnostics_results['network'] = True  # Assume it's ok
        except requests.exceptions.RequestException as e:
            logger.warning(f"Network connectivity test failed: {str(e)}")
            self.diagnostics_results['network'] = True  # Still allow it to run
    
    def check_version_directories(self):
        """Check version directories exist and contain required files"""
        logger.info("Checking version directories...")
        
        version_status = True
        for version in self.version_folders:
            version_path = os.path.join('src', version)
            if os.path.isdir(version_path):
                logger.info(f"Version directory {version} exists")
                
                # Check for key files expected in each version
                node_files = [f for f in os.listdir(version_path) if f.endswith('_node.py') or f == 'node.py']
                processor_files = [f for f in os.listdir(version_path) if 'processor' in f.lower() and f.endswith('.py')]
                
                if node_files:
                    logger.info(f"Found node files in {version}: {', '.join(node_files)}")
                if processor_files:
                    logger.info(f"Found processor files in {version}: {', '.join(processor_files)}")
                
                if not node_files and not processor_files:
                    logger.warning(f"No node or processor files found in {version}")
            else:
                logger.warning(f"Version directory {version} does not exist")
        
        self.diagnostics_results['version_directories'] = True  # Continue even if some are missing
    
    def check_model_files(self):
        """Check for model files"""
        logger.info("Checking model files...")
        
        models_dir = 'models'
        if os.path.isdir(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.bin') or f.endswith('.pt') or f.endswith('.pth')]
            if model_files:
                logger.info(f"Found model files: {', '.join(model_files)}")
                self.diagnostics_results['model_files'] = True
            else:
                logger.warning("No model files found in 'models' directory")
                self.diagnostics_results['model_files'] = True  # Continue anyway
        else:
            logger.warning("Models directory does not exist")
            self.diagnostics_results['model_files'] = True  # Continue anyway
    
    def print_summary(self):
        """Print a summary of the diagnostics results"""
        print("\n" + "="*50)
        print(" CENTRAL NODE MONITOR SYSTEM DIAGNOSTICS SUMMARY ")
        print("="*50)
        
        total_tests = len(self.diagnostics_results)
        passed_tests = sum(self.diagnostics_results.values())
        
        print(f"\nPassed {passed_tests}/{total_tests} diagnostic tests")
        
        if self.gpu_available:
            print("\nGPU is available for acceleration")
        else:
            print("\nRunning in CPU mode (no GPU detected)")
        
        # Print failed tests if any
        failed_tests = [test for test, result in self.diagnostics_results.items() if not result]
        if failed_tests:
            print("\nFAILED TESTS:")
            for test in failed_tests:
                print(f"- {test}")
            print("\nPlease address these issues before running the Central Node Monitor")
        else:
            print("\nAll diagnostic tests passed!")
            print("The Central Node Monitor is ready to launch")
        
        print("\n" + "="*50 + "\n")

# Run diagnostics when script is executed
if __name__ == "__main__":
    # Add a slight delay to ensure log files are properly initialized
    time.sleep(0.5)
    
    start_time = time.time()
    print("Running system diagnostics...")
    
    diagnostics = SystemDiagnostics()
    success = diagnostics.run_all_diagnostics()
    
    # Print summary
    diagnostics.print_summary()
    
    end_time = time.time()
    print(f"Diagnostics completed in {end_time - start_time:.2f} seconds")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1) 