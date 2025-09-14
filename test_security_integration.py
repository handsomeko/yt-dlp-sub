#!/usr/bin/env python3
"""
Security Integration Test Suite
Tests all integrated security measures in the FastAPI application
"""

import asyncio
import requests
import time
import json
from typing import Dict, List, Any
from datetime import datetime
import sys
import os

# Test configuration
API_BASE_URL = "http://127.0.0.1:8000"
VALID_API_KEY = "test-api-key-123"
INVALID_API_KEY = "invalid-key-456"
READONLY_API_KEY = "read-only-key-456"

class SecurityTestSuite:
    def __init__(self):
        self.passed_tests = 0
        self.failed_tests = 0
        self.test_results = []
    
    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Log test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"   {details}")
        
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
        
        self.test_results.append({
            "test": test_name,
            "passed": passed,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })
    
    def test_server_running(self) -> bool:
        """Test if secure server is running"""
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                security_active = data.get("security_active", False)
                self.log_test("Server Running", True, f"Status: {response.status_code}")
                self.log_test("Security Active", security_active, f"Security flag: {security_active}")
                return True
            else:
                self.log_test("Server Running", False, f"Unexpected status: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Server Running", False, f"Connection failed: {e}")
            return False
    
    def test_authentication_required(self):
        """Test that endpoints require authentication"""
        protected_endpoints = [
            "/video/info",
            "/video/download", 
            "/audio/download",
            "/transcript/extract",
            "/channel/add",
            "/channel/list"
        ]
        
        for endpoint in protected_endpoints:
            try:
                if endpoint in ["/video/info", "/audio/download"]:
                    # GET endpoints
                    response = requests.post(f"{API_BASE_URL}{endpoint}", 
                                           json={"url": "https://youtube.com/watch?v=test"},
                                           timeout=5)
                elif endpoint == "/channel/list":
                    # GET endpoint
                    response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=5)
                else:
                    # POST endpoints
                    response = requests.post(f"{API_BASE_URL}{endpoint}", 
                                           json={"test": "data"}, 
                                           timeout=5)
                
                if response.status_code == 401:
                    self.log_test(f"Auth Required: {endpoint}", True, "401 Unauthorized returned")
                else:
                    self.log_test(f"Auth Required: {endpoint}", False, 
                                f"Expected 401, got {response.status_code}")
            except Exception as e:
                self.log_test(f"Auth Required: {endpoint}", False, f"Request failed: {e}")
    
    def test_valid_authentication(self):
        """Test that valid API keys work"""
        headers = {"Authorization": f"Bearer {VALID_API_KEY}"}
        
        try:
            response = requests.get(f"{API_BASE_URL}/video/formats", 
                                  headers=headers, timeout=5)
            
            if response.status_code == 200:
                self.log_test("Valid API Key", True, "Access granted with valid key")
            else:
                self.log_test("Valid API Key", False, 
                            f"Expected 200, got {response.status_code}")
        except Exception as e:
            self.log_test("Valid API Key", False, f"Request failed: {e}")
    
    def test_invalid_authentication(self):
        """Test that invalid API keys are rejected"""
        headers = {"Authorization": f"Bearer {INVALID_API_KEY}"}
        
        try:
            response = requests.get(f"{API_BASE_URL}/video/formats", 
                                  headers=headers, timeout=5)
            
            if response.status_code == 401:
                self.log_test("Invalid API Key", True, "Invalid key properly rejected")
            else:
                self.log_test("Invalid API Key", False, 
                            f"Expected 401, got {response.status_code}")
        except Exception as e:
            self.log_test("Invalid API Key", False, f"Request failed: {e}")
    
    def test_permission_system(self):
        """Test that permission system works"""
        readonly_headers = {"Authorization": f"Bearer {READONLY_API_KEY}"}
        
        # Test read permission works
        try:
            response = requests.get(f"{API_BASE_URL}/video/formats", 
                                  headers=readonly_headers, timeout=5)
            
            if response.status_code == 200:
                self.log_test("Read Permission", True, "Read-only key can access read endpoints")
            else:
                self.log_test("Read Permission", False, f"Expected 200, got {response.status_code}")
        except Exception as e:
            self.log_test("Read Permission", False, f"Request failed: {e}")
        
        # Test write permission denied
        try:
            response = requests.post(f"{API_BASE_URL}/channel/add",
                                   json={"channel_id": "UC123", "channel_name": "Test"},
                                   headers=readonly_headers, timeout=5)
            
            if response.status_code == 403:
                self.log_test("Write Permission Denied", True, "Write access properly denied")
            else:
                self.log_test("Write Permission Denied", False, 
                            f"Expected 403, got {response.status_code}")
        except Exception as e:
            self.log_test("Write Permission Denied", False, f"Request failed: {e}")
    
    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        headers = {"Authorization": f"Bearer {VALID_API_KEY}"}
        
        print("🔄 Testing rate limiting (may take 60+ seconds)...")
        
        # Make rapid requests to trigger rate limiting
        success_count = 0
        rate_limited_count = 0
        
        for i in range(110):  # Exceed the 100 requests limit
            try:
                response = requests.get(f"{API_BASE_URL}/video/formats", 
                                      headers=headers, timeout=2)
                
                if response.status_code == 200:
                    success_count += 1
                elif response.status_code == 429:
                    rate_limited_count += 1
                    break  # Stop once we hit rate limit
                
                # Small delay between requests
                time.sleep(0.01)
            except Exception:
                continue
        
        if rate_limited_count > 0:
            self.log_test("Rate Limiting", True, 
                        f"Rate limit triggered after {success_count} requests")
        else:
            self.log_test("Rate Limiting", False, 
                        f"Rate limit not triggered after {success_count} requests")
    
    def test_input_validation(self):
        """Test input validation and sanitization"""
        headers = {"Authorization": f"Bearer {VALID_API_KEY}"}
        
        # Test malicious URL
        malicious_payloads = [
            "javascript:alert('xss')",
            "../../../etc/passwd",
            "https://evil.com/malicious.php?cmd=ls",
            "file:///etc/passwd",
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "$(rm -rf /)",
            "`whoami`"
        ]
        
        for payload in malicious_payloads:
            try:
                response = requests.post(f"{API_BASE_URL}/video/info",
                                       json={"url": payload},
                                       headers=headers, timeout=5)
                
                if response.status_code == 400:
                    self.log_test(f"Input Validation: {payload[:20]}...", True, 
                                "Malicious input properly rejected")
                else:
                    self.log_test(f"Input Validation: {payload[:20]}...", False,
                                f"Expected 400, got {response.status_code}")
            except Exception as e:
                self.log_test(f"Input Validation: {payload[:20]}...", False, 
                            f"Request failed: {e}")
    
    def test_security_headers(self):
        """Test that security headers are present"""
        headers = {"Authorization": f"Bearer {VALID_API_KEY}"}
        
        try:
            response = requests.get(f"{API_BASE_URL}/video/formats", 
                                  headers=headers, timeout=5)
            
            required_headers = {
                "X-XSS-Protection": "1; mode=block",
                "X-Content-Type-Options": "nosniff", 
                "X-Frame-Options": "DENY",
                "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
                "Content-Security-Policy": "default-src 'self'",
                "Referrer-Policy": "strict-origin-when-cross-origin",
                "Server": "Secure-Server"
            }
            
            missing_headers = []
            for header, expected_value in required_headers.items():
                if header not in response.headers:
                    missing_headers.append(header)
                elif expected_value not in response.headers[header]:
                    missing_headers.append(f"{header} (wrong value)")
            
            if not missing_headers:
                self.log_test("Security Headers", True, "All required headers present")
            else:
                self.log_test("Security Headers", False, 
                            f"Missing headers: {', '.join(missing_headers)}")
                
        except Exception as e:
            self.log_test("Security Headers", False, f"Request failed: {e}")
    
    def test_cors_protection(self):
        """Test CORS configuration"""
        # Test that requests from unauthorized origins are blocked
        headers = {
            "Authorization": f"Bearer {VALID_API_KEY}",
            "Origin": "https://evil-site.com"
        }
        
        try:
            response = requests.get(f"{API_BASE_URL}/video/formats", 
                                  headers=headers, timeout=5)
            
            # Check if CORS headers restrict the origin
            access_control_allow_origin = response.headers.get("Access-Control-Allow-Origin")
            
            if access_control_allow_origin and "evil-site.com" not in access_control_allow_origin:
                self.log_test("CORS Protection", True, "Evil origin not allowed")
            elif access_control_allow_origin == "*":
                self.log_test("CORS Protection", False, "Wildcard CORS allows all origins")
            else:
                self.log_test("CORS Protection", True, "CORS properly configured")
                
        except Exception as e:
            self.log_test("CORS Protection", False, f"Request failed: {e}")
    
    def test_error_handling(self):
        """Test secure error handling"""
        headers = {"Authorization": f"Bearer {VALID_API_KEY}"}
        
        try:
            # Test with invalid data to trigger error
            response = requests.post(f"{API_BASE_URL}/video/download",
                                   json={"url": "invalid-url", "quality": "invalid"},
                                   headers=headers, timeout=5)
            
            if response.status_code >= 400:
                error_data = response.json()
                
                # Check that error doesn't expose sensitive information
                sensitive_keywords = ["traceback", "file path", "database", "password", "secret"]
                error_text = str(error_data).lower()
                
                exposed_info = [keyword for keyword in sensitive_keywords if keyword in error_text]
                
                if not exposed_info:
                    self.log_test("Error Handling", True, "No sensitive info in errors")
                else:
                    self.log_test("Error Handling", False, 
                                f"Sensitive info exposed: {exposed_info}")
            else:
                self.log_test("Error Handling", False, "Error response expected but not received")
                
        except Exception as e:
            self.log_test("Error Handling", False, f"Request failed: {e}")
    
    def run_all_tests(self):
        """Run complete security test suite"""
        print("🔒 Starting Security Integration Test Suite")
        print("=" * 60)
        
        # Check if server is running
        if not self.test_server_running():
            print("\n❌ Cannot run tests - secure server not running!")
            print("Please start the secure server with:")
            print("cd /Users/jk/yt-dl-sub")
            print("python api/main_secure_integrated.py")
            return False
        
        print("\n🧪 Running Security Tests...")
        
        # Run all security tests
        self.test_authentication_required()
        self.test_valid_authentication()
        self.test_invalid_authentication()
        self.test_permission_system()
        self.test_rate_limiting()
        self.test_input_validation()
        self.test_security_headers()
        self.test_cors_protection()
        self.test_error_handling()
        
        # Print summary
        print("\n" + "=" * 60)
        print("🏁 Security Test Results Summary")
        print("=" * 60)
        
        total_tests = self.passed_tests + self.failed_tests
        pass_rate = (self.passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"✅ Passed: {self.passed_tests}")
        print(f"❌ Failed: {self.failed_tests}")
        print(f"📊 Pass Rate: {pass_rate:.1f}%")
        
        if self.failed_tests == 0:
            print("\n🎉 ALL SECURITY TESTS PASSED!")
            print("The integrated security measures are working correctly.")
        else:
            print(f"\n⚠️  {self.failed_tests} TESTS FAILED")
            print("Please review and fix the failing security measures.")
        
        # Save detailed results
        results_file = "security_test_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "summary": {
                    "total_tests": total_tests,
                    "passed": self.passed_tests,
                    "failed": self.failed_tests,
                    "pass_rate": pass_rate,
                    "timestamp": datetime.now().isoformat()
                },
                "detailed_results": self.test_results
            }, f, indent=2)
        
        print(f"\n📄 Detailed results saved to: {results_file}")
        
        return self.failed_tests == 0

def main():
    """Main function"""
    print("🔐 Security Integration Test Suite v1.0")
    
    # Check if we're in the right directory
    if not os.path.exists("api/main_secure_integrated.py"):
        print("❌ Error: Please run this script from the project root directory")
        print("Expected file not found: api/main_secure_integrated.py")
        return 1
    
    # Run tests
    test_suite = SecurityTestSuite()
    success = test_suite.run_all_tests()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())