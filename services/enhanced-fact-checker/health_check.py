#!/usr/bin/env python3
"""
V4 Enhanced Fact-Checker Health Check
"""

import requests
import sys

def health_check():
    """Simple health check for V4 fact-checking service"""
    try:
        response = requests.get("http://localhost:8885/health", timeout=5)
        if response.status_code == 200:
            print("✅ V4 Fact-Checker is healthy")
            return True
        else:
            print(f"❌ V4 Fact-Checker returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ V4 Fact-Checker health check failed: {str(e)}")
        return False

if __name__ == "__main__":
    if health_check():
        sys.exit(0)
    else:
        sys.exit(1) 