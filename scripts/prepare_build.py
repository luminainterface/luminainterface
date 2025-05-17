import os
import shutil
from pathlib import Path

def copy_lumina_core():
    # Get the project root directory
    root_dir = Path(__file__).parent.parent
    lumina_core_dir = root_dir / 'lumina_core'
    
    # List of service directories that need lumina-core
    services = [
        'concept-analyzer',
        'batch-embedder',
        'crawler',
        'concept-dictionary',
        'rag-coordinator'  # Added rag-coordinator
    ]
    
    # Copy lumina-core to each service directory
    for service in services:
        service_dir = root_dir / 'services' / service
        target_dir = service_dir / 'lumina-core'
        
        # Remove existing copy if it exists
        if target_dir.exists():
            shutil.rmtree(target_dir)
            
        # Copy lumina-core
        shutil.copytree(lumina_core_dir, target_dir)
        print(f"Copied lumina-core to {service}")

if __name__ == '__main__':
    copy_lumina_core() 