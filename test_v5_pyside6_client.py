#!/usr/bin/env python3
"""
Test Script for V5 PySide6 Client

This script runs the V5 PySide6 client in mock mode for testing without backend services.
"""

import os
import sys
import argparse
from pathlib import Path

def main():
    """Main entry point"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test V5 PySide6 Client")
    parser.add_argument("--theme", choices=["light", "dark", "system"], default="system", help="UI theme")
    parser.add_argument("--no-plugins", action="store_true", help="Disable plugin discovery")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    # Set up command-line arguments for the client
    client_args = ["--mock"]  # Always use mock mode for testing
    
    if args.theme != "system":
        client_args.extend(["--theme", args.theme])
    
    if args.no_plugins:
        client_args.append("--no-plugins")
    
    if args.debug:
        client_args.append("--debug")
    
    # Run the client with the specified arguments
    command = [sys.executable, "v5_pyside6_client.py"] + client_args
    
    print(f"Running: {' '.join(command)}")
    os.execvp(command[0], command)

if __name__ == "__main__":
    main() 