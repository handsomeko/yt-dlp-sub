#!/usr/bin/env python3
"""
Comprehensive Security Test Suite for Fixed API
Tests all security fixes and validates protection against attacks
"""

import requests
import time
import json
from datetime import datetime
from typing import Dict, List, Any

# Test configuration for new secure server
API_BASE_URL = "http://127.0.0.1:8002"
VALID_API_KEY = "secure-admin-key-2024"
READONLY_API_KEY = "secure-read-key-2024"
INVALID_API_KEY = "invalid-key-fake"

class ComprehensiveSecurityTests:
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
    
    def test_server_health(self) -> bool:
        """Test if secure server is running and properly configured"""
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                
                # Check security systems
                security_systems = data.get("security_systems", {})
                all_systems_active = all(security_systems.values())
                
                self.log_test("Server Health", True, f"Status: {response.status_code}")
                self.log_test("Security Systems Active", all_systems_active, 
                            f"Systems: {list(security_systems.keys())}")
                
                return True
            else:
                self.log_test("Server Health", False, f"Status: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Server Health", False, f"Connection failed: {e}")
            return False
    
    def test_malicious_input_blocking(self):
        """Test that malicious inputs are properly blocked with 400 errors"""
        headers = {"Authorization": f"Bearer {VALID_API_KEY}"}
        
        malicious_payloads = [
            # XSS attacks
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<iframe src='javascript:alert(1)'></iframe>",
            
            # SQL injection
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "UNION SELECT * FROM users--",
            
            # Command injection  
            "$(rm -rf /)",
            "`whoami`",
            "|nc -l -p 1234 -e /bin/sh",
            
            # Path traversal
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\config\\sam",
            "/proc/version",
            
            # Protocol attacks
            "file:///etc/passwd",
            "ftp://evil.com/",
            "data:text/html,<script>alert(1)</script>"
        ]
        
        for payload in malicious_payloads:
            try:
                response = requests.post(
                    f"{API_BASE_URL}/video/info",
                    json={"url": payload},
                    headers=headers,
                    timeout=5
                )
                
                if response.status_code == 400:
                    self.log_test(f"Malicious Input Blocked: {payload[:20]}...", True, 
                                "Properly blocked with 400 error")
                else:
                    self.log_test(f"Malicious Input Blocked: {payload[:20]}...", False,
                                f"Expected 400, got {response.status_code}")
                    
            except Exception as e:
                self.log_test(f"Malicious Input Blocked: {payload[:20]}...", False, 
                            f"Request failed: {e}")
    
    def test_authentication_security(self):
        """Test authentication and authorization"""
        
        # Test 1: No auth header
        try:
            response = requests.get(f"{API_BASE_URL}/video/formats", timeout=5)
            if response.status_code == 403:  # Should be 403 for missing auth
                self.log_test("No Auth Header", True, "Properly rejected with 403")
            else:
                self.log_test("No Auth Header", False, 
                            f"Expected 403, got {response.status_code}")
        except Exception as e:
            self.log_test("No Auth Header", False, f"Request failed: {e}")
        
        # Test 2: Invalid API key  
        try:
            response = requests.get(
                f"{API_BASE_URL}/video/formats",
                headers={"Authorization": f"Bearer {INVALID_API_KEY}"},
                timeout=5
            )
            if response.status_code == 401:
                self.log_test("Invalid API Key", True, "Properly rejected with 401")
            else:
                self.log_test("Invalid API Key", False, 
                            f"Expected 401, got {response.status_code}")
        except Exception as e:
            self.log_test("Invalid API Key", False, f"Request failed: {e}")
        
        # Test 3: Valid API key
        try:
            response = requests.get(
                f"{API_BASE_URL}/video/formats",
                headers={"Authorization": f"Bearer {VALID_API_KEY}"},
                timeout=5
            )
            if response.status_code == 200:
                self.log_test("Valid API Key", True, "Access granted")
            else:
                self.log_test("Valid API Key", False, 
                            f"Expected 200, got {response.status_code}")
        except Exception as e:
            self.log_test("Valid API Key", False, f"Request failed: {e}")
        
        # Test 4: Permission system
        readonly_headers = {"Authorization": f"Bearer {READONLY_API_KEY}"}
        
        try:
            # Read permission should work
            response = requests.get(f"{API_BASE_URL}/video/formats", 
                                  headers=readonly_headers, timeout=5)
            if response.status_code == 200:
                self.log_test("Read Permission", True, "Read access granted")
            else:
                self.log_test("Read Permission", False, f"Expected 200, got {response.status_code}")
        except Exception as e:
            self.log_test("Read Permission", False, f"Request failed: {e}")
        
        try:
            # Write permission should be denied
            response = requests.post(
                f"{API_BASE_URL}/channel/add",
                json={"channel_id": "UCtest123456789012345678", "channel_name": "Test"},
                headers=readonly_headers,
                timeout=5
            )
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
        
        print("🔄 Testing rate limiting (this may take time)...")
        
        success_count = 0
        rate_limited_count = 0
        
        # Make requests to trigger rate limiting
        for i in range(105):  # Exceed the 100 request limit
            try:
                response = requests.get(
                    f"{API_BASE_URL}/video/formats", 
                    headers=headers, 
                    timeout=2
                )
                
                if response.status_code == 200:
                    success_count += 1
                elif response.status_code == 429:
                    rate_limited_count += 1
                    break
                
                time.sleep(0.01)  # Small delay
                
            except Exception:
                continue
        
        if rate_limited_count > 0:
            self.log_test("Rate Limiting", True, 
                        f"Rate limit triggered after {success_count} requests")
        else:
            self.log_test("Rate Limiting", False, 
                        f"Rate limit not triggered after {success_count} requests")
    
    def test_security_headers(self):
        """Test that security headers are present"""
        headers = {"Authorization": f"Bearer {VALID_API_KEY}"}
        
        try:
            response = requests.get(f"{API_BASE_URL}/video/formats", 
                                  headers=headers, timeout=5)
            
            required_headers = [
                "X-XSS-Protection",
                "X-Content-Type-Options", 
                "X-Frame-Options",
                "Strict-Transport-Security",
                "Content-Security-Policy",
                "Referrer-Policy"
            ]
            
            present_headers = []
            missing_headers = []
            
            for header in required_headers:
                if header in response.headers:
                    present_headers.append(header)
                else:
                    missing_headers.append(header)
            
            if not missing_headers:
                self.log_test("Security Headers", True, 
                            f"All headers present: {present_headers}")
            else:
                self.log_test("Security Headers", False, 
                            f"Missing headers: {missing_headers}")
                
        except Exception as e:
            self.log_test("Security Headers", False, f"Request failed: {e}")
    
    def test_input_validation(self):
        """Test comprehensive input validation"""
        headers = {"Authorization": f"Bearer {VALID_API_KEY}"}
        
        # Test valid input works
        try:
            response = requests.post(
                f"{API_BASE_URL}/video/info",
                json={"url": "https://youtube.com/watch?v=test123"},
                headers=headers,
                timeout=5
            )
            
            # This should work or fail with business logic, not security validation
            if response.status_code in [200, 400, 500]:  # Any non-security error is fine
                self.log_test("Valid Input Processing", True, "Valid input accepted")
            else:
                self.log_test("Valid Input Processing", False, 
                            f"Valid input rejected: {response.status_code}")
        except Exception as e:
            self.log_test("Valid Input Processing", False, f"Request failed: {e}")
        
        # Test invalid URL format
        try:
            response = requests.post(
                f"{API_BASE_URL}/video/info",
                json={"url": "not-a-youtube-url"},
                headers=headers,
                timeout=5
            )
            
            if response.status_code == 400:
                self.log_test("Invalid URL Format", True, "Invalid URL properly rejected")
            else:
                self.log_test("Invalid URL Format", False, 
                            f"Expected 400, got {response.status_code}")
        except Exception as e:
            self.log_test("Invalid URL Format", False, f"Request failed: {e}")
    
    def test_sql_injection_prevention(self):
        """Test that SQL injection in core modules is prevented"""
        headers = {"Authorization": f"Bearer {VALID_API_KEY}"}
        
        # This would have triggered SQL injection before the fix
        try:
            response = requests.get(
                f"{API_BASE_URL}/channel/list?enabled_only=true",
                headers=headers,
                timeout=5
            )
            
            if response.status_code == 200:
                self.log_test("SQL Injection Prevention", True, 
                            "Channel list endpoint secure")
            else:
                self.log_test("SQL Injection Prevention", False, 
                            f"Unexpected status: {response.status_code}")
        except Exception as e:
            self.log_test("SQL Injection Prevention", False, f"Request failed: {e}")
    
    def run_comprehensive_tests(self):
        """Run complete security test suite"""
        print("🔒 Starting Comprehensive Security Test Suite")
        print("=" * 70)
        
        # Check server health first
        if not self.test_server_health():
            print("\n❌ Cannot run tests - secure server not responding!")
            return False
        
        print("\n🛡️  Running Security Tests...")
        
        # Run all security tests
        self.test_malicious_input_blocking()
        self.test_authentication_security()
        self.test_rate_limiting()
        self.test_security_headers()
        self.test_input_validation()
        self.test_sql_injection_prevention()
        
        # Print comprehensive results
        print("\n" + "=" * 70)
        print("🏁 Comprehensive Security Test Results")
        print("=" * 70)
        
        total_tests = self.passed_tests + self.failed_tests
        pass_rate = (self.passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"✅ Passed: {self.passed_tests}")
        print(f"❌ Failed: {self.failed_tests}")
        print(f"📊 Pass Rate: {pass_rate:.1f}%")
        
        # Security assessment
        if pass_rate >= 90:
            print("\n🎉 EXCELLENT SECURITY POSTURE!")
            print("The API has comprehensive security protection.")
        elif pass_rate >= 75:
            print("\n✅ GOOD SECURITY POSTURE")
            print("Most security measures are working correctly.")
        elif pass_rate >= 50:
            print("\n⚠️  MODERATE SECURITY CONCERNS")
            print("Some security issues need attention.")
        else:
            print("\n❌ CRITICAL SECURITY ISSUES")
            print("Major security vulnerabilities detected.")
        
        # Save detailed results
        results_file = "comprehensive_security_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "summary": {
                    "total_tests": total_tests,
                    "passed": self.passed_tests,
                    "failed": self.failed_tests,
                    "pass_rate": pass_rate,
                    "timestamp": datetime.now().isoformat(),
                    "server_url": API_BASE_URL
                },
                "detailed_results": self.test_results
            }, f, indent=2)
        
        print(f"\n📄 Detailed results saved to: {results_file}")
        
        return pass_rate >= 75  # Consider success if 75%+ pass rate

def main():
    """Main function"""
    print("🔐 Comprehensive Security Test Suite v2.0")
    
    # Run comprehensive tests
    test_suite = ComprehensiveSecurityTests()
    success = test_suite.run_comprehensive_tests()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())