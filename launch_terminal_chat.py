#!/usr/bin/env python3
"""
TERMINAL CHAT INTERFACE LAUNCHER - CONFIDENCE-DRIVEN LORA SYSTEM
===============================================================
Quick launcher for the terminal chat interface with Docker system integration.
"""

import subprocess
import sys
import time
import os
import json
from pathlib import Path

def check_docker_status():
    """Check if Docker and docker-compose are available"""
    try:
        # Check Docker
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Docker not found or not running")
            return False
        print(f"✅ Docker: {result.stdout.strip()}")
        
        # Check Docker Compose
        result = subprocess.run(['docker', 'compose', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Docker Compose not found")
            return False
        print(f"✅ Docker Compose: {result.stdout.strip()}")
        
        return True
    except Exception as e:
        print(f"❌ Docker check failed: {e}")
        return False

def start_confidence_system():
    """Start the confidence-driven LoRA system"""
    print("\n🧠💡 Starting Confidence-Driven LoRA System...")
    print("=" * 60)
    
    # Check if docker-compose file exists
    if not Path("docker-compose-v10-ultimate.yml").exists():
        print("❌ docker-compose-v10-ultimate.yml not found!")
        return False
    
    try:
        # Start only the confidence-driven services first
        confidence_services = [
            'confidence-driven-lora-creator',
            'ultimate-chat-orchestrator-with-confidence', 
            'high-rank-adapter',
            'enhanced-crawler-nlp',
            'concept-training-worker'
        ]
        
        print("🚀 Starting core confidence services...")
        for service in confidence_services:
            print(f"  Starting {service}...")
            result = subprocess.run([
                'docker', 'compose', '-f', 'docker-compose-v10-ultimate.yml', 
                'up', '-d', service
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"  ✅ {service} started")
            else:
                print(f"  ⚠️  {service} had issues: {result.stderr[:100]}")
        
        # Wait a moment for services to start
        print("\n⏳ Waiting for services to initialize...")
        time.sleep(15)
        
        # Check service health
        print("\n🏥 Checking service health...")
        healthy_services = check_service_health()
        
        if healthy_services >= 3:
            print(f"✅ System ready with {healthy_services} services!")
            return True
        else:
            print(f"⚠️  Only {healthy_services} services are healthy. Proceeding with available services...")
            return True
            
    except Exception as e:
        print(f"❌ Failed to start system: {e}")
        return False

def check_service_health():
    """Check health of running services"""
    import requests
    
    services = {
        'HiRa Adapter': 'http://localhost:9000/health',
        'Ultimate Chat': 'http://localhost:8950/health',
        'Confidence Creator': 'http://localhost:8848/health',
        'Enhanced Crawler': 'http://localhost:8850/health',
        'Concept Training': 'http://localhost:8851/health'
    }
    
    healthy_count = 0
    for service_name, url in services.items():
        try:
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                print(f"  ✅ {service_name}")
                healthy_count += 1
            else:
                print(f"  ⚠️  {service_name} (Status: {response.status_code})")
        except Exception as e:
            print(f"  ❌ {service_name} (Not responding)")
    
    return healthy_count

def install_dependencies():
    """Install required Python dependencies"""
    print("\n📦 Installing Python dependencies...")
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', 
            'colorama', 'aiohttp', 'requests', 'asyncio'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
        else:
            print(f"⚠️  Some dependencies may not have installed: {result.stderr[:100]}")
    except Exception as e:
        print(f"❌ Failed to install dependencies: {e}")

def show_welcome_message():
    """Show welcome message and instructions"""
    print("\n" + "="*70)
    print("🧠💡 CONFIDENCE-DRIVEN LORA SYSTEM - TERMINAL CHAT INTERFACE")
    print("="*70)
    print("This terminal interface connects to the entire confidence-driven system")
    print("for testing and validation with real-time metrics and HiRa steering.")
    print("\n🎯 FEATURES:")
    print("  • Real-time confidence monitoring")
    print("  • Automatic knowledge gap detection")
    print("  • HiRa (High-Rank Adapter) steering")
    print("  • Live metrics dashboard")
    print("  • Comprehensive system diagnostics")
    print("  • Session reporting and analytics")
    print("\n📋 COMMANDS:")
    print("  /metrics     - Show live metrics dashboard")
    print("  /diagnostics - Run system diagnostics")
    print("  /quit        - Exit the chat interface")
    print("\n🚀 Ready to test the confidence-driven learning system!")
    print("="*70)

def main():
    """Main launcher function"""
    show_welcome_message()
    
    # Check prerequisites
    print("\n🔍 Checking prerequisites...")
    if not check_docker_status():
        print("❌ Docker is required. Please install Docker and try again.")
        sys.exit(1)
    
    # Install dependencies
    install_dependencies()
    
    # Ask user if they want to start the system
    print(f"\n🤔 Do you want to start the confidence-driven system? (y/n): ", end="")
    start_system = input().lower().strip()
    
    if start_system in ['y', 'yes', '']:
        if start_confidence_system():
            print(f"\n🎉 System started successfully!")
        else:
            print(f"\n⚠️  System startup had issues, but proceeding...")
    else:
        print("📝 Skipping system startup. Make sure services are running manually.")
    
    # Launch terminal interface
    print(f"\n🚀 Launching terminal chat interface...")
    print("(Press Ctrl+C to exit)")
    
    try:
        # Import and run the terminal interface
        from terminal_chat_system_interface import main as chat_main
        import asyncio
        asyncio.run(chat_main())
        
    except ImportError:
        print("❌ Terminal chat interface not found. Running directly...")
        try:
            subprocess.run([sys.executable, 'terminal_chat_system_interface.py'])
        except Exception as e:
            print(f"❌ Failed to launch interface: {e}")
    except Exception as e:
        print(f"❌ Interface error: {e}")
    
    print("\n👋 Thanks for testing the confidence-driven system!")

if __name__ == "__main__":
    main() 