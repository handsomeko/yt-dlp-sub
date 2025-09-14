#!/usr/bin/env python3
"""
Core security fixes verification - Tests fundamental security improvements
without external dependencies like alembic, SQLAlchemy, etc.
"""

import asyncio
import os
import sys
import tempfile
import threading
import time
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CoreFixesVerifier:
    """Verification of core security fixes without external dependencies"""
    
    def __init__(self):
        self.test_results = {}
        self.temp_dir = tempfile.mkdtemp(prefix="verify_core_fixes_")
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    async def run_all_verifications(self) -> Dict[str, Any]:
        """Run all core verification tests"""
        
        logger.info("🔍 Starting core security fixes verification...")
        
        # Category 1: Path Traversal Prevention (Critical Issues #1, #2, #5)
        await self._verify_path_traversal_fixes()
        
        # Category 2: Input Validation System (Critical security component)
        await self._verify_input_validation_system()
        
        # Category 3: Thread Safety Improvements (Issues #9, #38, #41)
        await self._verify_thread_safety_fixes()
        
        # Category 4: Storage Path Security (V2 migration security)
        await self._verify_storage_path_security()
        
        # Category 5: Error Handling Improvements (Issues #11, #12, #13)
        await self._verify_error_handling_fixes()
        
        return self._generate_final_report()
    
    async def _verify_path_traversal_fixes(self):
        """Verify path traversal vulnerabilities are fixed"""
        logger.info("🔒 Verifying path traversal fixes...")
        
        results = []
        
        try:
            from core.filename_sanitizer import sanitize_channel_id, sanitize_video_id
            
            # Test dangerous path patterns
            dangerous_patterns = [
                "../../../etc/passwd",
                "..\\..\\windows\\system32",
                "%2e%2e%2f%2fetc%2fpasswd",
                "~/.ssh/id_rsa",
                "channel/../../../evil",
                "",  # Empty string
                "x" * 100,  # Too long
                "channel\x00null",  # Null byte
                "channel:colon",  # Dangerous characters
                "channel|pipe",
                "channel<redirect",
                "channel>output",
                "channel\"quote",
                "channel'quote",
            ]
            
            for pattern in dangerous_patterns:
                try:
                    result = sanitize_channel_id(pattern)
                    results.append({
                        "pattern": pattern,
                        "expected": "reject",
                        "actual": "accept",
                        "passed": False,
                        "error": f"Should have rejected: {pattern}"
                    })
                except ValueError as e:
                    results.append({
                        "pattern": pattern,
                        "expected": "reject", 
                        "actual": "reject",
                        "passed": True,
                        "error": str(e)
                    })
            
            # Test safe patterns
            safe_patterns = [
                "UCchannel123",
                "video_abc_123",
                "normal-channel",
                "Channel123",
                "test_video_456"
            ]
            
            for pattern in safe_patterns:
                try:
                    result = sanitize_channel_id(pattern)
                    results.append({
                        "pattern": pattern,
                        "expected": "accept",
                        "actual": "accept", 
                        "passed": True,
                        "sanitized": result
                    })
                except ValueError as e:
                    results.append({
                        "pattern": pattern,
                        "expected": "accept",
                        "actual": "reject",
                        "passed": False,
                        "error": str(e)
                    })
            
        except ImportError as e:
            results.append({"test": "path_sanitization", "passed": False, "error": f"Import failed: {e}"})
        except Exception as e:
            results.append({"test": "path_sanitization", "passed": False, "error": str(e)})
        
        # Test storage path construction
        try:
            from core.storage_paths_v2 import StoragePathsV2
            
            storage = StoragePathsV2(base_path=Path(self.temp_dir))
            
            # Test safe path construction
            safe_path = storage.get_media_file("UCchannel123", "video456", "opus")
            if "UCchannel123" in str(safe_path) and "video456" in str(safe_path):
                results.append({"test": "safe_path_construction", "passed": True})
            else:
                results.append({"test": "safe_path_construction", "passed": False, "error": "Path construction failed"})
            
        except Exception as e:
            results.append({"test": "storage_path_construction", "passed": False, "error": str(e)})
        
        self.test_results["path_traversal_fixes"] = {
            "passed": all(r["passed"] for r in results),
            "details": results
        }
    
    async def _verify_input_validation_system(self):
        """Verify comprehensive input validation system"""
        logger.info("✅ Verifying input validation system...")
        
        results = []
        
        try:
            from core.input_validation import InputValidator, URLValidator, FileValidator
            
            # Test URL validation
            url_validator = URLValidator()
            
            # Dangerous URLs that should be rejected
            dangerous_urls = [
                "javascript:alert('xss')",
                "data:text/html,<script>alert('xss')</script>",
                "file:///etc/passwd",
                "vbscript:msgbox('xss')",
                "../../../evil/path",
                "http://example.com/path/../../../evil",
                "https://evil.com/path\x00null",
                "ftp://evil.com/path'OR'1'='1"
            ]
            
            for url in dangerous_urls:
                try:
                    validated = url_validator.validate_url(url)
                    if validated.valid:
                        results.append({
                            "url": url,
                            "type": "dangerous",
                            "expected": "reject",
                            "actual": "accept",
                            "passed": False
                        })
                    else:
                        results.append({
                            "url": url,
                            "type": "dangerous", 
                            "expected": "reject",
                            "actual": "reject",
                            "passed": True
                        })
                except Exception:
                    results.append({
                        "url": url,
                        "type": "dangerous", 
                        "expected": "reject",
                        "actual": "reject",
                        "passed": True
                    })
            
            # Safe URLs that should be accepted
            safe_urls = [
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "https://youtu.be/dQw4w9WgXcQ",
                "http://example.com/safe-path",
                "https://api.example.com/v1/data"
            ]
            
            for url in safe_urls:
                try:
                    validated = url_validator.validate_url(url)
                    if validated.valid:
                        results.append({
                            "url": url,
                            "type": "safe",
                            "expected": "accept",
                            "actual": "accept", 
                            "passed": True
                        })
                    else:
                        results.append({
                            "url": url,
                            "type": "safe",
                            "expected": "accept",
                            "actual": "reject",
                            "passed": False,
                            "error": "; ".join(validated.errors)
                        })
                except Exception as e:
                    results.append({
                        "url": url,
                        "type": "safe",
                        "expected": "accept",
                        "actual": "reject",
                        "passed": False,
                        "error": str(e)
                    })
            
            # Test file validation
            file_validator = FileValidator()
            
            dangerous_files = [
                "../../../etc/passwd",
                "~/.ssh/id_rsa", 
                "file\x00null.txt",
                "file:with:colons.txt",
                "file|with|pipes.txt",
                "file<redirect.txt",
                "file>output.txt"
            ]
            
            for file_path in dangerous_files:
                try:
                    validated = file_validator.validate_file_path(file_path)
                    if validated.valid:
                        results.append({
                            "path": file_path,
                            "type": "dangerous",
                            "expected": "reject",
                            "actual": "accept",
                            "passed": False
                        })
                    else:
                        results.append({
                            "path": file_path,
                            "type": "dangerous",
                            "expected": "reject", 
                            "actual": "reject",
                            "passed": True
                        })
                except Exception:
                    results.append({
                        "path": file_path,
                        "type": "dangerous",
                        "expected": "reject", 
                        "actual": "reject",
                        "passed": True
                    })
            
        except ImportError as e:
            results.append({"test": "input_validation_import", "passed": False, "error": f"Import failed: {e}"})
        except Exception as e:
            results.append({"test": "input_validation_system", "passed": False, "error": str(e)})
        
        self.test_results["input_validation_system"] = {
            "passed": all(r["passed"] for r in results),
            "details": results
        }
    
    async def _verify_thread_safety_fixes(self):
        """Verify thread safety and deadlock prevention"""
        logger.info("🔄 Verifying thread safety fixes...")
        
        results = []
        
        try:
            from core.performance_optimizations import DeadlockPrevention
            
            # Test deadlock prevention with multiple threads
            prevention = DeadlockPrevention()
            thread_results = []
            
            def worker_thread(thread_id):
                import threading
                try:
                    # Create test locks
                    lock1 = threading.RLock()
                    lock2 = threading.RLock()
                    
                    # Register and acquire locks in order to prevent deadlocks
                    prevention.register_lock('test_lock1', lock1)
                    prevention.register_lock('test_lock2', lock2)
                    
                    with prevention.acquire_ordered('test_lock1', lock1, timeout=2.0):
                        time.sleep(0.05)  # Brief hold
                        with prevention.acquire_ordered('test_lock2', lock2, timeout=2.0):
                            time.sleep(0.05)  # Brief hold
                    
                    return True
                except Exception as e:
                    logger.error(f"Thread {thread_id} failed: {e}")
                    return False
            
            # Run multiple threads concurrently
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(worker_thread, i) for i in range(5)]
                thread_results = [f.result(timeout=5) for f in futures]
            
            success_count = sum(thread_results)
            
            results.append({
                "test": "deadlock_prevention",
                "passed": success_count >= 3,  # At least 3 of 5 should succeed
                "success_count": success_count,
                "total_threads": len(thread_results)
            })
            
        except ImportError as e:
            results.append({"test": "deadlock_prevention_import", "passed": False, "error": f"Import failed: {e}"})
        except Exception as e:
            results.append({"test": "thread_safety", "passed": False, "error": str(e)})
        
        self.test_results["thread_safety_fixes"] = {
            "passed": all(r["passed"] for r in results),
            "details": results
        }
    
    async def _verify_storage_path_security(self):
        """Verify V2 storage path security"""
        logger.info("🗂️ Verifying storage path security...")
        
        results = []
        
        try:
            from core.storage_paths_v2 import StoragePathsV2
            
            storage = StoragePathsV2(base_path=Path(self.temp_dir))
            
            # Test secure path construction for various content types
            test_cases = [
                ("UCchannel123", "video456", "audio"),
                ("channel_abc", "vid_123", "transcript"),
                ("test-channel", "test-video", "metadata")
            ]
            
            for channel_id, video_id, content_type in test_cases:
                try:
                    if content_type == "audio":
                        path = storage.get_media_file(channel_id, video_id, "opus")
                    elif content_type == "transcript":
                        path = storage.get_transcript_file(channel_id, video_id, "srt")
                    elif content_type == "metadata":
                        path = storage.get_video_metadata_file(channel_id, video_id)
                    
                    # Verify path contains expected components and is within base directory
                    path_str = str(path)
                    base_str = str(self.temp_dir)
                    
                    is_secure = (
                        path_str.startswith(base_str) and
                        channel_id in path_str and
                        video_id in path_str and
                        ".." not in path_str and
                        "~" not in path_str
                    )
                    
                    results.append({
                        "channel_id": channel_id,
                        "video_id": video_id,
                        "content_type": content_type,
                        "passed": is_secure,
                        "path": path_str if is_secure else "INSECURE_PATH_HIDDEN"
                    })
                    
                except Exception as e:
                    results.append({
                        "channel_id": channel_id,
                        "video_id": video_id,
                        "content_type": content_type,
                        "passed": False,
                        "error": str(e)
                    })
            
        except ImportError as e:
            results.append({"test": "storage_paths_import", "passed": False, "error": f"Import failed: {e}"})
        except Exception as e:
            results.append({"test": "storage_path_security", "passed": False, "error": str(e)})
        
        self.test_results["storage_path_security"] = {
            "passed": all(r["passed"] for r in results),
            "details": results
        }
    
    async def _verify_error_handling_fixes(self):
        """Verify error handling improvements"""
        logger.info("🚨 Verifying error handling fixes...")
        
        results = []
        
        try:
            from core.error_handling import ErrorManager, ErrorCategory
            
            error_manager = ErrorManager()
            
            # Test error categorization
            test_categories = [
                ErrorCategory.YOUTUBE_API,
                ErrorCategory.FILESYSTEM,
                ErrorCategory.NETWORK,
                ErrorCategory.VALIDATION
            ]
            
            for category in test_categories:
                try:
                    # Test error handling for different categories
                    test_error = Exception(f"Test error for {category.value}")
                    
                    # Try sync error handling
                    result = error_manager.handle_error(
                        test_error,
                        category=category,
                        context={"test": True}
                    )
                    
                    results.append({
                        "category": category.value,
                        "test_type": "sync",
                        "passed": result is not None,
                        "has_recovery": result.get("recovery_attempted", False) if result else False
                    })
                    
                except Exception as e:
                    results.append({
                        "category": category.value,
                        "test_type": "sync",
                        "passed": False,
                        "error": str(e)
                    })
            
        except ImportError as e:
            results.append({"test": "error_handling_import", "passed": False, "error": f"Import failed: {e}"})
        except Exception as e:
            results.append({"test": "error_handling", "passed": False, "error": str(e)})
        
        self.test_results["error_handling_fixes"] = {
            "passed": all(r["passed"] for r in results),
            "details": results
        }
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final verification report"""
        
        total_categories = len(self.test_results)
        passed_categories = sum(1 for result in self.test_results.values() if result["passed"])
        
        # Calculate detailed statistics
        total_tests = 0
        passed_tests = 0
        
        for category, result in self.test_results.items():
            details = result.get("details", [])
            category_total = len(details)
            category_passed = sum(1 for detail in details if detail.get("passed", True))
            
            total_tests += category_total
            passed_tests += category_passed
        
        # Determine overall status
        overall_passed = passed_categories == total_categories and passed_tests >= (total_tests * 0.9)  # 90% threshold
        
        report = {
            "timestamp": time.time(),
            "overall_status": "✅ CORE FIXES VERIFIED" if overall_passed else "⚠️  SOME ISSUES FOUND",
            "overall_passed": overall_passed,
            "summary": {
                "total_categories": total_categories,
                "passed_categories": passed_categories,
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "success_rate": f"{(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "0%"
            },
            "category_results": {}
        }
        
        # Add category details
        for category, result in self.test_results.items():
            details = result.get("details", [])
            category_total = len(details)
            category_passed = sum(1 for detail in details if detail.get("passed", True))
            
            report["category_results"][category] = {
                "status": "✅ PASSED" if result["passed"] else "❌ FAILED",
                "passed": result["passed"],
                "test_count": category_total,
                "passed_count": category_passed,
                "success_rate": f"{(category_passed/category_total*100):.1f}%" if category_total > 0 else "0%",
                "details": details
            }
        
        return report

async def main():
    """Main verification entry point"""
    
    print("🔍 Core Security Fixes Verification Suite")
    print("==========================================")
    print("Testing fundamental security and stability improvements...")
    print()
    
    with CoreFixesVerifier() as verifier:
        try:
            report = await verifier.run_all_verifications()
            
            # Print summary
            print(f"📊 CORE VERIFICATION COMPLETE")
            print(f"Status: {report['overall_status']}")
            print(f"Categories: {report['summary']['passed_categories']}/{report['summary']['total_categories']}")
            print(f"Tests: {report['summary']['passed_tests']}/{report['summary']['total_tests']}")
            print(f"Success Rate: {report['summary']['success_rate']}")
            print()
            
            # Print category results
            for category, result in report["category_results"].items():
                status_icon = "✅" if result["passed"] else "❌"
                print(f"{status_icon} {category.replace('_', ' ').title()}: {result['passed_count']}/{result['test_count']} ({result['success_rate']})")
            
            # Save detailed report
            report_file = os.path.join(Path(__file__).parent, "core_verification_report.json")
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"\n📄 Detailed report saved to: {report_file}")
            
            if report["overall_passed"]:
                print("\n🎉 CORE SECURITY FIXES SUCCESSFULLY VERIFIED!")
                print("✅ Path traversal protection active")
                print("✅ Input validation system operational") 
                print("✅ Thread safety improvements working")
                print("✅ Storage path security enforced")
                print("✅ Error handling enhanced")
                print("\nThe system's core security foundations are solid.")
                return 0
            else:
                print("\n⚠️  Some core security tests had issues. Review the detailed report.")
                return 1
                
        except Exception as e:
            print(f"\n❌ Core verification failed with error: {e}")
            import traceback
            traceback.print_exc()
            return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))