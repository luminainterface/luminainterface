#!/usr/bin/env python3
"""
Enhanced A2A Coordination Hub - Deployment & Management Script
Handles building, deployment, testing, and monitoring of the intelligent orchestrator
"""

import subprocess
import sys
import time
import json
import os
import argparse
from pathlib import Path

class EnhancedOrchestratorDeployer:
    def __init__(self):
        self.service_name = "enhanced-a2a-coordination-hub"
        self.container_name = "enhanced-a2a-orchestrator"
        self.port = 8891
        self.image_tag = "enhanced-a2a-hub:latest"
        
    def run_command(self, command: str, capture_output: bool = False):
        """Execute shell command with proper error handling"""
        print(f"🔧 Executing: {command}")
        try:
            if capture_output:
                result = subprocess.run(command, shell=True, capture_output=True, text=True)
                return result.stdout.strip() if result.returncode == 0 else None
            else:
                result = subprocess.run(command, shell=True, check=True)
                return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Command failed: {e}")
            return False
    
    def build_image(self):
        """Build the enhanced orchestrator Docker image"""
        print("🏗️ Building Enhanced A2A Coordination Hub Docker Image...")
        
        # Check if Dockerfile exists
        if not Path("Dockerfile").exists():
            print("❌ Dockerfile not found!")
            return False
        
        # Build the image
        build_command = f"docker build -t {self.image_tag} ."
        return self.run_command(build_command)
    
    def stop_existing_container(self):
        """Stop and remove existing container if running"""
        print("🛑 Stopping existing container...")
        
        # Check if container exists and is running
        check_command = f"docker ps -q -f name={self.container_name}"
        container_id = self.run_command(check_command, capture_output=True)
        
        if container_id:
            print(f"📦 Found running container: {container_id}")
            self.run_command(f"docker stop {self.container_name}")
            self.run_command(f"docker rm {self.container_name}")
        else:
            print("✅ No existing container found")
    
    def start_container(self, detached: bool = True):
        """Start the enhanced orchestrator container"""
        print("🚀 Starting Enhanced A2A Coordination Hub...")
        
        detach_flag = "-d" if detached else ""
        run_command = (
            f"docker run {detach_flag} "
            f"--name {self.container_name} "
            f"-p {self.port}:{self.port} "
            f"--restart unless-stopped "
            f"{self.image_tag}"
        )
        
        return self.run_command(run_command)
    
    def check_health(self, max_attempts: int = 10):
        """Check if the service is healthy and ready"""
        print("🏥 Checking service health...")
        
        import requests
        
        for attempt in range(max_attempts):
            try:
                response = requests.get(f"http://localhost:{self.port}/health", timeout=5)
                if response.status_code == 200:
                    health_data = response.json()
                    print(f"✅ Service is healthy: {health_data}")
                    return True
            except Exception as e:
                print(f"⏳ Attempt {attempt + 1}/{max_attempts} - Health check failed: {e}")
            
            time.sleep(3)
        
        print("❌ Service health check failed after all attempts")
        return False
    
    def run_tests(self):
        """Run the comprehensive test suite"""
        print("🧪 Running Enhanced Orchestrator Test Suite...")
        
        # Check if test file exists
        if not Path("test_enhanced_orchestrator.py").exists():
            print("❌ Test file not found!")
            return False
        
        # Run tests
        test_command = "python test_enhanced_orchestrator.py"
        return self.run_command(test_command)
    
    def show_logs(self, follow: bool = False):
        """Show container logs"""
        follow_flag = "-f" if follow else ""
        logs_command = f"docker logs {follow_flag} {self.container_name}"
        self.run_command(logs_command)
    
    def show_status(self):
        """Show detailed service status"""
        print("📊 Enhanced A2A Coordination Hub Status")
        print("=" * 50)
        
        # Container status
        status_command = f"docker ps -f name={self.container_name}"
        print("🐳 Container Status:")
        self.run_command(status_command)
        
        # Service health
        try:
            import requests
            response = requests.get(f"http://localhost:{self.port}/health", timeout=5)
            print(f"\n🏥 Health Check: {response.json()}")
            
            # Get metrics
            metrics_response = requests.get(f"http://localhost:{self.port}/metrics", timeout=5)
            print(f"\n📈 Metrics: {metrics_response.json()}")
            
        except Exception as e:
            print(f"❌ Could not retrieve service status: {e}")
    
    def deploy_full(self):
        """Complete deployment workflow"""
        print("🚀 ENHANCED A2A COORDINATION HUB - FULL DEPLOYMENT")
        print("=" * 60)
        
        steps = [
            ("🏗️ Building image", self.build_image),
            ("🛑 Stopping existing container", self.stop_existing_container),
            ("🚀 Starting new container", lambda: self.start_container(detached=True)),
            ("🏥 Health check", lambda: self.check_health(max_attempts=15)),
            ("🧪 Running tests", self.run_tests),
            ("📊 Final status", self.show_status)
        ]
        
        for step_name, step_func in steps:
            print(f"\n{step_name}...")
            if not step_func():
                print(f"❌ {step_name} failed! Deployment aborted.")
                return False
        
        print("\n" + "=" * 60)
        print("🎉 ENHANCED A2A COORDINATION HUB DEPLOYED SUCCESSFULLY!")
        print(f"🌐 Service available at: http://localhost:{self.port}")
        print("🔧 Features enabled:")
        print("  ✅ Confidence-driven orchestration")
        print("  ✅ Mathematical validation with SymPy")
        print("  ✅ 18+ microservice integration")
        print("  ✅ Multi-agent coordination")
        print("  ✅ Real-time health monitoring")
        return True

def main():
    parser = argparse.ArgumentParser(description="Enhanced A2A Coordination Hub Deployment")
    parser.add_argument("action", choices=[
        "build", "start", "stop", "restart", "test", "logs", "status", "deploy"
    ], help="Action to perform")
    parser.add_argument("--follow", action="store_true", help="Follow logs (for logs action)")
    
    args = parser.parse_args()
    deployer = EnhancedOrchestratorDeployer()
    
    if args.action == "build":
        deployer.build_image()
    elif args.action == "start":
        deployer.start_container()
    elif args.action == "stop":
        deployer.stop_existing_container()
    elif args.action == "restart":
        deployer.stop_existing_container()
        deployer.start_container()
    elif args.action == "test":
        deployer.run_tests()
    elif args.action == "logs":
        deployer.show_logs(follow=args.follow)
    elif args.action == "status":
        deployer.show_status()
    elif args.action == "deploy":
        deployer.deploy_full()

if __name__ == "__main__":
    main() 