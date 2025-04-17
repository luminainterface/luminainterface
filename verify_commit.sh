#!/bin/bash

echo "=== Git Repository Verification ==="
echo

# Check git status
echo "1. Checking git status..."
git status
echo

# Check commit history
echo "2. Checking commit history..."
git log --oneline -n 1
echo

# Check tracked files
echo "3. Checking tracked files..."
git ls-files | wc -l
echo "Number of files tracked: $(git ls-files | wc -l)"
echo

# Check remote setup
echo "4. Checking remote repository..."
git remote -v
echo

# Check branch
echo "5. Checking current branch..."
git branch
echo

# Check specific directories
echo "6. Verifying key directories..."
echo "Version directories:"
ls -d src/v*/ 2>/dev/null || echo "Version directories not found"
echo
echo "GitHub workflows:"
ls -d .github/workflows/*.yml 2>/dev/null || echo "Workflow files not found"
echo

# Check file contents
echo "7. Verifying key files..."
echo "Version bridge files:"
ls -l src/version_bridge*.py 2>/dev/null || echo "Version bridge files not found"
echo
echo "Core files:"
ls -l src/{main.py,__init__.py,central_node.py,central_node_monitor.py} 2>/dev/null || echo "Core files not found"
echo

echo "=== Verification Complete ==="
echo "If you see any 'not found' messages, those files/directories need to be added."
echo "Make sure all the numbers match what you expect from the initial commit." 