#!/usr/bin/env python3
"""Quick script to set up a test API key for the secure API"""

import sys
sys.path.append('/Users/jk/yt-dl-sub')

from core.critical_security_fixes_v2 import AuthenticationSystem

# Create authentication system
auth = AuthenticationSystem()

# Generate a test API key
test_key = auth.generate_api_key(
    user_id="test_user", 
    permissions=["download", "transcript", "quality_check"]
)

print(f"Test API key generated: {test_key}")
print("\nUse this key in the Authorization header:")
print(f"Authorization: Bearer {test_key}")