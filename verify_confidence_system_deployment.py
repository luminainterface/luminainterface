#!/usr/bin/env python3
"""
CONFIDENCE-DRIVEN SYSTEM DEPLOYMENT VERIFICATION
===============================================
Comprehensive verification script to ensure all components of the 
confidence-driven LoRA system are properly configured and ready for deployment.
"""

import os
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import subprocess

class ConfidenceSystemVerifier:
    """Verifies all components of the confidence-driven system"""
    
    def __init__(self):
        self.base_path = Path.cwd()
        self.verification_results = {
            'files': {},
            'dockerfiles': {},
            'directories': {},
            'dependencies': {},
            'docker_compose': {},
            'overall_status': 'pending'
        }
        
        # Core files required for confidence-driven system
        self.required_files = {
            'confidence_driven_lora_creator.py': 'Core confidence-driven LoRA creator service',
            'ultimate_chat_orchestrator_with_confidence.py': 'Enhanced chat orchestrator with confidence monitoring',
            'test_confidence_driven_system.py': 'Comprehensive testing suite',
            'deploy_confidence_driven_system.py': 'Deployment automation script',
            'confidence_demo.py': 'Standalone demonstration system',
            'docker-compose-v10-ultimate.yml': 'Docker Compose configuration',
            'requirements.txt': 'Python dependencies'
        }
        
        # Required Dockerfiles
        self.required_dockerfiles = {
            'Dockerfile.confidence-driven-lora': 'Confidence-driven LoRA creator Dockerfile',
            'Dockerfile.ultimate-chat-confidence': 'Ultimate chat orchestrator Dockerfile',
            'Dockerfile.confidence-system-tester': 'Testing system Dockerfile',
            'Dockerfile.confidence-system-deployer': 'Deployment system Dockerfile',
            'Dockerfile.confidence-demo': 'Demo service Dockerfile'
        }
        
        # Required directories
        self.required_directories = [
            'logs/confidence_lora',
            'logs/chat_orchestrator',
            'logs/confidence_testing',
            'logs/deployment',
            'logs/confidence_demo',
            'knowledge_gaps',
            'confidence_patterns',
            'lora_requests',
            'conversations',
            'confidence_sessions',
            'user_analytics',
            'test_results/confidence_system',
            'reports/confidence_system',
            'deployment_reports',
            'demo_results'
        ]
        
        # Docker Compose services to verify
        self.confidence_services = [
            'confidence-driven-lora-creator',
            'ultimate-chat-orchestrator-with-confidence',
            'confidence-driven-system-tester',
            'confidence-driven-system-deployer',
            'confidence-demo-service'
        ]
    
    def run_verification(self) -> bool:
        """Run complete verification of the confidence-driven system"""
        
        print("🧠💡 CONFIDENCE-DRIVEN SYSTEM DEPLOYMENT VERIFICATION")
        print("=" * 60)
        
        success = True
        
        # Verify files
        print("\n📄 Verifying Core Files...")
        files_ok = self._verify_files()
        success = success and files_ok
        
        # Verify Dockerfiles
        print("\n🐳 Verifying Dockerfiles...")
        dockerfiles_ok = self._verify_dockerfiles()
        success = success and dockerfiles_ok
        
        # Verify directories
        print("\n📁 Verifying Directories...")
        directories_ok = self._verify_directories()
        success = success and directories_ok
        
        # Verify dependencies
        print("\n📦 Verifying Dependencies...")
        dependencies_ok = self._verify_dependencies()
        success = success and dependencies_ok
        
        # Verify Docker Compose configuration
        print("\n🔧 Verifying Docker Compose Configuration...")
        compose_ok = self._verify_docker_compose()
        success = success and compose_ok
        
        # Generate verification report
        self.verification_results['overall_status'] = 'passed' if success else 'failed'
        self._generate_verification_report()
        
        return success
    
    def _verify_files(self) -> bool:
        """Verify all required files exist and are valid"""
        
        all_files_ok = True
        
        for filename, description in self.required_files.items():
            file_path = self.base_path / filename
            
            if file_path.exists():
                # Check file size and basic content
                file_size = file_path.stat().st_size
                if file_size > 100:  # Must be substantial file
                    print(f"  ✅ {filename} - {description} ({file_size} bytes)")
                    self.verification_results['files'][filename] = {
                        'status': 'found',
                        'size': file_size,
                        'description': description
                    }
                else:
                    print(f"  ⚠️  {filename} - Found but suspiciously small ({file_size} bytes)")
                    self.verification_results['files'][filename] = {
                        'status': 'found_small',
                        'size': file_size,
                        'description': description
                    }
                    all_files_ok = False
            else:
                print(f"  ❌ {filename} - MISSING")
                self.verification_results['files'][filename] = {
                    'status': 'missing',
                    'description': description
                }
                all_files_ok = False
        
        return all_files_ok
    
    def _verify_dockerfiles(self) -> bool:
        """Verify all required Dockerfiles exist and have proper content"""
        
        all_dockerfiles_ok = True
        
        for dockerfile, description in self.required_dockerfiles.items():
            dockerfile_path = self.base_path / dockerfile
            
            if dockerfile_path.exists():
                # Read and verify Dockerfile content
                with open(dockerfile_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                has_from = 'FROM' in content
                has_workdir = 'WORKDIR' in content
                has_copy = 'COPY' in content or 'ADD' in content
                has_cmd = 'CMD' in content or 'ENTRYPOINT' in content
                
                if has_from and has_workdir and has_copy and has_cmd:
                    print(f"  ✅ {dockerfile} - {description} (Valid Dockerfile)")
                    self.verification_results['dockerfiles'][dockerfile] = {
                        'status': 'valid',
                        'description': description,
                        'size': len(content)
                    }
                else:
                    print(f"  ⚠️  {dockerfile} - Found but missing required sections")
                    self.verification_results['dockerfiles'][dockerfile] = {
                        'status': 'incomplete',
                        'description': description,
                        'missing': [
                            k for k, v in {
                                'FROM': has_from,
                                'WORKDIR': has_workdir,
                                'COPY/ADD': has_copy,
                                'CMD/ENTRYPOINT': has_cmd
                            }.items() if not v
                        ]
                    }
                    all_dockerfiles_ok = False
            else:
                print(f"  ❌ {dockerfile} - MISSING")
                self.verification_results['dockerfiles'][dockerfile] = {
                    'status': 'missing',
                    'description': description
                }
                all_dockerfiles_ok = False
        
        return all_dockerfiles_ok
    
    def _verify_directories(self) -> bool:
        """Verify all required directories exist"""
        
        all_directories_ok = True
        
        for directory in self.required_directories:
            dir_path = self.base_path / directory
            
            if dir_path.exists() and dir_path.is_dir():
                print(f"  ✅ {directory}")
                self.verification_results['directories'][directory] = {
                    'status': 'exists',
                    'path': str(dir_path)
                }
            else:
                print(f"  ❌ {directory} - MISSING")
                self.verification_results['directories'][directory] = {
                    'status': 'missing',
                    'path': str(dir_path)
                }
                all_directories_ok = False
        
        return all_directories_ok
    
    def _verify_dependencies(self) -> bool:
        """Verify requirements.txt has all necessary dependencies"""
        
        requirements_path = self.base_path / 'requirements.txt'
        
        if not requirements_path.exists():
            print("  ❌ requirements.txt not found")
            self.verification_results['dependencies']['status'] = 'missing'
            return False
        
        with open(requirements_path, 'r', encoding='utf-8') as f:
            requirements_content = f.read()
        
        # Key dependencies for confidence-driven system
        required_packages = [
            'fastapi',
            'uvicorn',
            'pydantic',
            'aiohttp',
            'requests',
            'redis',
            'transformers',
            'torch',
            'sentence-transformers',
            'qdrant-client',
            'neo4j',
            'sympy',
            'asyncio-throttle',
            'pytest',
            'docker'
        ]
        
        missing_packages = []
        found_packages = []
        
        for package in required_packages:
            if package.lower() in requirements_content.lower():
                found_packages.append(package)
            else:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"  ⚠️  Missing packages: {', '.join(missing_packages)}")
            self.verification_results['dependencies'] = {
                'status': 'incomplete',
                'found': found_packages,
                'missing': missing_packages,
                'total_lines': len(requirements_content.splitlines())
            }
            return False
        else:
            print(f"  ✅ All required packages found ({len(found_packages)} packages)")
            self.verification_results['dependencies'] = {
                'status': 'complete',
                'found': found_packages,
                'total_lines': len(requirements_content.splitlines())
            }
            return True
    
    def _verify_docker_compose(self) -> bool:
        """Verify Docker Compose configuration includes confidence services"""
        
        compose_path = self.base_path / 'docker-compose-v10-ultimate.yml'
        
        if not compose_path.exists():
            print("  ❌ docker-compose-v10-ultimate.yml not found")
            self.verification_results['docker_compose']['status'] = 'missing'
            return False
        
        with open(compose_path, 'r', encoding='utf-8') as f:
            compose_content = f.read()
        
        missing_services = []
        found_services = []
        
        for service in self.confidence_services:
            if service in compose_content:
                found_services.append(service)
            else:
                missing_services.append(service)
        
        # Check for confidence-related configurations
        has_confidence_volumes = 'confidence_' in compose_content
        has_confidence_ports = '8848:8848' in compose_content and '8950:8950' in compose_content
        has_confidence_networks = 'ultimate-network' in compose_content
        
        if missing_services:
            print(f"  ⚠️  Missing services: {', '.join(missing_services)}")
            self.verification_results['docker_compose'] = {
                'status': 'incomplete',
                'found_services': found_services,
                'missing_services': missing_services,
                'has_volumes': has_confidence_volumes,
                'has_ports': has_confidence_ports,
                'has_networks': has_confidence_networks
            }
            return False
        else:
            print(f"  ✅ All confidence services found ({len(found_services)} services)")
            self.verification_results['docker_compose'] = {
                'status': 'complete',
                'found_services': found_services,
                'has_volumes': has_confidence_volumes,
                'has_ports': has_confidence_ports,
                'has_networks': has_confidence_networks
            }
            return True
    
    def _generate_verification_report(self):
        """Generate a comprehensive verification report"""
        
        report_path = self.base_path / 'deployment_reports' / 'confidence_system_verification.json'
        
        # Ensure report directory exists
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add summary statistics
        self.verification_results['summary'] = {
            'files_verified': len([f for f in self.verification_results['files'].values() if f['status'] == 'found']),
            'files_total': len(self.required_files),
            'dockerfiles_verified': len([d for d in self.verification_results['dockerfiles'].values() if d['status'] == 'valid']),
            'dockerfiles_total': len(self.required_dockerfiles),
            'directories_verified': len([d for d in self.verification_results['directories'].values() if d['status'] == 'exists']),
            'directories_total': len(self.required_directories),
            'dependencies_complete': self.verification_results['dependencies'].get('status') == 'complete',
            'docker_compose_complete': self.verification_results['docker_compose'].get('status') == 'complete'
        }
        
        # Write detailed report
        with open(report_path, 'w') as f:
            json.dump(self.verification_results, f, indent=2)
        
        print(f"\n📊 Verification Report Generated: {report_path}")
        
        # Print summary
        summary = self.verification_results['summary']
        print("\n📋 VERIFICATION SUMMARY:")
        print(f"  Files: {summary['files_verified']}/{summary['files_total']}")
        print(f"  Dockerfiles: {summary['dockerfiles_verified']}/{summary['dockerfiles_total']}")
        print(f"  Directories: {summary['directories_verified']}/{summary['directories_total']}")
        print(f"  Dependencies: {'✅' if summary['dependencies_complete'] else '❌'}")
        print(f"  Docker Compose: {'✅' if summary['docker_compose_complete'] else '❌'}")
        
        overall_success = (
            summary['files_verified'] == summary['files_total'] and
            summary['dockerfiles_verified'] == summary['dockerfiles_total'] and
            summary['directories_verified'] == summary['directories_total'] and
            summary['dependencies_complete'] and
            summary['docker_compose_complete']
        )
        
        if overall_success:
            print("\n🎉 CONFIDENCE-DRIVEN SYSTEM READY FOR DEPLOYMENT!")
        else:
            print("\n⚠️  CONFIDENCE-DRIVEN SYSTEM NEEDS ATTENTION BEFORE DEPLOYMENT")
        
        return overall_success

def main():
    """Main verification function"""
    
    verifier = ConfidenceSystemVerifier()
    success = verifier.run_verification()
    
    if success:
        print("\n🚀 DEPLOYMENT COMMANDS:")
        print("docker compose -f docker-compose-v10-ultimate.yml up -d")
        print("docker compose -f docker-compose-v10-ultimate.yml --profile confidence-testing run --rm confidence-driven-system-tester")
        print("docker compose -f docker-compose-v10-ultimate.yml --profile confidence-deployment run --rm confidence-driven-system-deployer")
        
        return 0
    else:
        print("\n❌ Please fix the issues above before deploying the confidence-driven system.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 