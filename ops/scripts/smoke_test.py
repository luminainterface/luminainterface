#!/usr/bin/env python3
"""
Lumina staging smoke test script.
Tests all available endpoints and functionality.
"""

import requests
import sys
import time
from typing import Dict, Any
import json

# Configuration
BASE_URL = "http://localhost:3000"  # Frontend
API_URL = "http://localhost:8000"   # Backend
TIMEOUT = 10  # seconds

def print_status(message: str, success: bool = True) -> None:
    """Print status message with color."""
    color = "\033[92m" if success else "\033[91m"  # Green for success, Red for failure
    reset = "\033[0m"
    print(f"{color}{'✓' if success else '✗'} {message}{reset}")

def test_frontend_assets() -> bool:
    """Test frontend static assets."""
    print("\nTesting Frontend Assets:")
    assets = [
        "/",
        "/css/styles.css",
        "/js/app.js",
        "/js/metrics.js",
        "/50x.html"
    ]
    
    success = True
    for asset in assets:
        try:
            response = requests.get(f"{BASE_URL}{asset}", timeout=TIMEOUT)
            if response.status_code == 200:
                print_status(f"Asset {asset} is accessible")
            else:
                print_status(f"Asset {asset} returned {response.status_code}", False)
                success = False
        except requests.RequestException as e:
            print_status(f"Failed to access {asset}: {str(e)}", False)
            success = False
    return success

def test_backend_endpoints() -> bool:
    """Test all backend API endpoints."""
    print("\nTesting Backend Endpoints:")
    
    # Test chat completion
    try:
        chat_data = {
            "model": "phi",
            "messages": [{"role": "user", "content": "Hello"}]
        }
        response = requests.post(
            f"{API_URL}/v1/chat/completions",
            json=chat_data,
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT
        )
        if response.status_code == 200:
            print_status("Chat completion endpoint is working")
        else:
            print_status(f"Chat completion returned {response.status_code}", False)
            return False
    except requests.RequestException as e:
        print_status(f"Chat completion failed: {str(e)}", False)
        return False

    # Test metrics endpoint
    try:
        response = requests.get(
            f"{API_URL}/metrics/summary",
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT
        )
        if response.status_code == 200:
            metrics = response.json()
            print_status("Metrics endpoint is working")
            print(f"  Cache size: {metrics.get('cache_size', 'N/A')}")
            print(f"  Cache hits: {metrics.get('cache_hits', 'N/A')}")
            print(f"  Cache misses: {metrics.get('cache_misses', 'N/A')}")
        else:
            print_status(f"Metrics endpoint returned {response.status_code}", False)
            return False
    except requests.RequestException as e:
        print_status(f"Metrics endpoint failed: {str(e)}", False)
        return False

    # Test admin prune endpoint
    try:
        response = requests.post(
            f"{API_URL}/admin/prune",
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT
        )
        if response.status_code == 200:
            print_status("Admin prune endpoint is working")
        else:
            print_status(f"Admin prune returned {response.status_code}", False)
            return False
    except requests.RequestException as e:
        print_status(f"Admin prune failed: {str(e)}", False)
        return False

    return True

def test_rate_limiting() -> bool:
    """Test rate limiting functionality."""
    print("\nTesting Rate Limiting:")
    
    # Test chat rate limiting
    try:
        chat_data = {
            "model": "phi",
            "messages": [{"role": "user", "content": "Hello"}]
        }
        
        # Make multiple requests rapidly
        responses = []
        for i in range(15):  # Increased number of requests
            response = requests.post(
                f"{API_URL}/v1/chat/completions",
                json=chat_data,
                headers={"Content-Type": "application/json"},
                timeout=TIMEOUT
            )
            responses.append(response)
            print_status(f"Request {i+1}: {response.status_code}")
            time.sleep(0.2)  # Reduced delay to trigger rate limit
        
        # Check if rate limiting kicked in
        rate_limited = any(r.status_code == 429 for r in responses)
        if rate_limited:
            print_status("Rate limiting is working")
            time.sleep(5)  # Add delay before next test
        else:
            print_status("Rate limiting not detected", False)
            return False
    except requests.RequestException as e:
        print_status(f"Rate limit test failed: {str(e)}", False)
        return False

    return True

def test_error_handling() -> bool:
    """Test error handling for various scenarios."""
    print("\nTesting Error Handling:")
    
    # Test invalid chat request
    try:
        invalid_data = {"invalid": "data"}
        response = requests.post(
            f"{API_URL}/v1/chat/completions",
            json=invalid_data,
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT
        )
        if response.status_code in [400, 422]:
            print_status("Invalid request handling is working")
        else:
            print_status(f"Invalid request returned {response.status_code}", False)
            return False
    except requests.RequestException as e:
        print_status(f"Invalid request test failed: {str(e)}", False)
        return False

    # Test non-existent endpoint
    try:
        response = requests.get(f"{API_URL}/nonexistent", timeout=TIMEOUT)
        if response.status_code == 404:
            print_status("404 handling is working")
        else:
            print_status(f"Nonexistent endpoint returned {response.status_code}", False)
            return False
    except requests.RequestException as e:
        print_status(f"404 test failed: {str(e)}", False)
        return False

    return True

def main() -> None:
    """Run all smoke tests."""
    print("Starting Lumina staging smoke tests...")
    
    tests = [
        ("Frontend Assets", test_frontend_assets),
        ("Backend Endpoints", test_backend_endpoints),
        ("Rate Limiting", test_rate_limiting),
        ("Error Handling", test_error_handling)
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        print(f"\n=== Testing {test_name} ===")
        try:
            if not test_func():
                all_passed = False
            time.sleep(3)  # Add delay between test suites
        except Exception as e:
            print_status(f"Test failed with error: {str(e)}", False)
            all_passed = False
    
    print("\n=== Test Summary ===")
    if all_passed:
        print_status("All tests passed!", True)
        sys.exit(0)
    else:
        print_status("Some tests failed!", False)
        sys.exit(1)

if __name__ == "__main__":
    main() 