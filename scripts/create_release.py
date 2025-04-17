#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path
import re
from datetime import datetime
import os

def get_project_root():
    """Get the project root directory"""
    current_dir = Path.cwd()
    while current_dir != current_dir.parent:
        if (current_dir / 'setup.py').exists():
            return current_dir
        current_dir = current_dir.parent
    raise ValueError("Could not find project root directory")

def get_current_version():
    """Get current version from setup.py"""
    root_dir = get_project_root()
    setup_path = root_dir / 'setup.py'
    
    with open(setup_path, 'r', encoding='utf-8') as f:
        content = f.read()
        match = re.search(r"version=['\"]([^'\"]+)['\"]", content)
        if match:
            return match.group(1)
        raise ValueError("Could not find version in setup.py")

def update_version(new_version):
    """Update version in setup.py"""
    root_dir = get_project_root()
    setup_path = root_dir / 'setup.py'
    
    with open(setup_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = re.sub(
        r"version=['\"]([^'\"]+)['\"]",
        f"version='{new_version}'",
        content
    )
    
    with open(setup_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

def update_changelog(version, changes):
    """Update CHANGELOG.md with new version"""
    root_dir = get_project_root()
    changelog_path = root_dir / 'CHANGELOG.md'
    
    if not changelog_path.exists():
        changelog_path.write_text("# Changelog\n\n", encoding='utf-8')
    
    with open(changelog_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    today = datetime.now().strftime("%Y-%m-%d")
    new_entry = f"\n## [{version}] - {today}\n\n{changes}\n"
    
    # Insert after the first line (after "# Changelog")
    lines = content.split('\n')
    lines.insert(1, new_entry)
    
    with open(changelog_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

def create_release(version, changes):
    """Create a new release"""
    try:
        root_dir = get_project_root()
        
        # Check if remote is configured
        result = subprocess.run(['git', 'remote', '-v'], 
                              capture_output=True, 
                              text=True, 
                              cwd=root_dir)
        
        if not result.stdout.strip():
            print("No git remote configured. Please configure a remote repository first.")
            print("You can do this by running:")
            print("git remote add origin <repository-url>")
            sys.exit(1)
        
        # Update version
        update_version(version)
        
        # Update changelog
        update_changelog(version, changes)
        
        # Commit changes
        subprocess.run(['git', 'add', str(root_dir / 'setup.py'), str(root_dir / 'CHANGELOG.md')], 
                      check=True, cwd=root_dir)
        subprocess.run(['git', 'commit', '-m', f'Bump version to {version}'], 
                      check=True, cwd=root_dir)
        
        # Create tag
        subprocess.run(['git', 'tag', f'v{version}'], check=True, cwd=root_dir)
        
        # Push changes
        subprocess.run(['git', 'push', 'origin', 'HEAD'], check=True, cwd=root_dir)
        subprocess.run(['git', 'push', 'origin', '--tags'], check=True, cwd=root_dir)
        
        print(f"Successfully created release v{version}")
        print("The CI/CD pipeline will now build and deploy the new version")
        
    except subprocess.CalledProcessError as e:
        print(f"Error during release creation: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Create a new release')
    parser.add_argument('version', help='New version number (e.g., 0.1.1)')
    parser.add_argument('--changes', help='Release notes/changes', required=True)
    
    args = parser.parse_args()
    
    current_version = get_current_version()
    print(f"Current version: {current_version}")
    print(f"New version: {args.version}")
    print(f"Changes:\n{args.changes}")
    
    confirm = input("\nProceed with release? (y/n): ")
    if confirm.lower() == 'y':
        create_release(args.version, args.changes)
    else:
        print("Release cancelled")

if __name__ == '__main__':
    main() 