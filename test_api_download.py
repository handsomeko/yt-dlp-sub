#!/usr/bin/env python3
"""Test the secure API with a YouTube URL"""

import requests
import json
import sys
sys.path.append('/Users/jk/yt-dl-sub')

from core.critical_security_fixes_v2 import AuthenticationSystem

# Create authentication system and generate API key
auth = AuthenticationSystem()
api_key = auth.generate_api_key(
    user_id="test_user", 
    permissions=["download", "transcript", "quality_check"]
)

print(f"Generated API key: {api_key}\n")

# Test URL from user
test_url = "https://www.youtube.com/watch?v=GT0jtVjRy2E"

# Make request to API
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

data = {
    "url": test_url,
    "quality": "1080p",
    "format": "mp4"
}

print(f"Testing video info endpoint with: {test_url}")
print("Making request to: http://127.0.0.1:8004/video/info\n")

try:
    response = requests.post(
        "http://127.0.0.1:8004/video/info",
        headers=headers,
        json=data,
        timeout=30
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")