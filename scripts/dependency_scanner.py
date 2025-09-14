#!/usr/bin/env python3
"""
Dependency Security Scanner - Issue #133
Automated vulnerability scanning for Python and npm dependencies
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import argparse
import os


class DependencySecurityScanner:
    """Comprehensive dependency vulnerability scanner"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.vulnerabilities = {}
        self.scan_timestamp = None
        self.report_dir = Path("security_reports")
        self.report_dir.mkdir(exist_ok=True)
    
    def log(self, message: str, level: str = "INFO"):
        """Log messages with level"""
        if self.verbose or level in ["ERROR", "WARNING"]:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] [{level}] {message}")
    
    def check_tool_availability(self) -> Dict[str, bool]:
        """Check if required security tools are installed"""
        tools = {
            "safety": "pip install safety",
            "pip-audit": "pip install pip-audit",
            "npm": "Install Node.js",
            "bandit": "pip install bandit",
            "semgrep": "pip install semgrep"
        }
        
        available = {}
        for tool, install_cmd in tools.items():
            try:
                if tool == "npm":
                    subprocess.run(["npm", "--version"], capture_output=True, check=True)
                else:
                    subprocess.run(["which", tool], capture_output=True, check=True)
                available[tool] = True
                self.log(f"✓ {tool} is available", "INFO")
            except subprocess.CalledProcessError:
                available[tool] = False
                self.log(f"✗ {tool} not found. Install with: {install_cmd}", "WARNING")
        
        return available
    
    def scan_python_with_safety(self) -> Dict[str, Any]:
        """Scan Python dependencies using safety"""
        self.log("Scanning Python dependencies with Safety...", "INFO")
        vulnerabilities = []
        
        try:
            # Generate requirements file from current environment
            req_result = subprocess.run(
                ["pip", "freeze"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Save temporary requirements
            temp_req = Path("/tmp/temp_requirements.txt")
            temp_req.write_text(req_result.stdout)
            
            # Run safety check
            result = subprocess.run(
                ["safety", "check", "--json", "-r", str(temp_req)],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.stdout:
                data = json.loads(result.stdout)
                for item in data:
                    vulnerabilities.append({
                        "package": item.get("package", "unknown"),
                        "installed_version": item.get("installed_version"),
                        "vulnerability_id": item.get("vulnerability_id"),
                        "severity": item.get("severity", "unknown"),
                        "description": item.get("description", ""),
                        "fixed_version": item.get("fixed_version")
                    })
            
            # Clean up
            temp_req.unlink(missing_ok=True)
            
        except subprocess.TimeoutExpired:
            self.log("Safety scan timeout", "ERROR")
        except json.JSONDecodeError as e:
            self.log(f"Failed to parse Safety output: {e}", "ERROR")
        except Exception as e:
            self.log(f"Safety scan failed: {e}", "ERROR")
        
        return {
            "tool": "safety",
            "scan_time": datetime.now().isoformat(),
            "vulnerabilities": vulnerabilities,
            "total": len(vulnerabilities)
        }
    
    def scan_python_with_pip_audit(self) -> Dict[str, Any]:
        """Scan Python dependencies using pip-audit"""
        self.log("Scanning Python dependencies with pip-audit...", "INFO")
        vulnerabilities = []
        
        try:
            result = subprocess.run(
                ["pip-audit", "--format", "json"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.stdout:
                data = json.loads(result.stdout)
                for dep in data.get("dependencies", []):
                    for vuln in dep.get("vulns", []):
                        vulnerabilities.append({
                            "package": dep.get("name"),
                            "installed_version": dep.get("version"),
                            "vulnerability_id": vuln.get("id"),
                            "severity": vuln.get("fix_versions", [""])[0] if vuln.get("fix_versions") else "unknown",
                            "description": vuln.get("description", ""),
                            "fixed_version": vuln.get("fix_versions", [""])[0] if vuln.get("fix_versions") else None
                        })
        
        except subprocess.TimeoutExpired:
            self.log("pip-audit scan timeout", "ERROR")
        except json.JSONDecodeError as e:
            self.log(f"Failed to parse pip-audit output: {e}", "ERROR")
        except Exception as e:
            self.log(f"pip-audit scan failed: {e}", "ERROR")
        
        return {
            "tool": "pip-audit",
            "scan_time": datetime.now().isoformat(),
            "vulnerabilities": vulnerabilities,
            "total": len(vulnerabilities)
        }
    
    def scan_npm_dependencies(self) -> Dict[str, Any]:
        """Scan npm dependencies using npm audit"""
        self.log("Scanning npm dependencies...", "INFO")
        vulnerabilities = []
        
        # Check if package.json exists
        if not Path("package.json").exists():
            self.log("No package.json found, skipping npm scan", "INFO")
            return {
                "tool": "npm-audit",
                "scan_time": datetime.now().isoformat(),
                "vulnerabilities": [],
                "total": 0
            }
        
        try:
            result = subprocess.run(
                ["npm", "audit", "--json"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.stdout:
                data = json.loads(result.stdout)
                
                # Parse npm audit output
                for advisory_id, advisory in data.get("advisories", {}).items():
                    vulnerabilities.append({
                        "package": advisory.get("module_name"),
                        "vulnerability_id": f"npm-{advisory_id}",
                        "severity": advisory.get("severity"),
                        "title": advisory.get("title"),
                        "description": advisory.get("overview", ""),
                        "recommendation": advisory.get("recommendation", ""),
                        "url": advisory.get("url", ""),
                        "paths": advisory.get("paths", [])
                    })
        
        except subprocess.TimeoutExpired:
            self.log("npm audit timeout", "ERROR")
        except json.JSONDecodeError as e:
            self.log(f"Failed to parse npm audit output: {e}", "ERROR")
        except Exception as e:
            self.log(f"npm audit failed: {e}", "ERROR")
        
        return {
            "tool": "npm-audit",
            "scan_time": datetime.now().isoformat(),
            "vulnerabilities": vulnerabilities,
            "total": len(vulnerabilities)
        }
    
    def scan_code_with_bandit(self) -> Dict[str, Any]:
        """Scan Python code for security issues using Bandit"""
        self.log("Scanning Python code with Bandit...", "INFO")
        issues = []
        
        try:
            result = subprocess.run(
                ["bandit", "-r", ".", "-f", "json", "-ll"],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.stdout:
                data = json.loads(result.stdout)
                
                for issue in data.get("results", []):
                    issues.append({
                        "file": issue.get("filename"),
                        "line": issue.get("line_number"),
                        "severity": issue.get("issue_severity"),
                        "confidence": issue.get("issue_confidence"),
                        "issue": issue.get("issue_text"),
                        "test_id": issue.get("test_id"),
                        "test_name": issue.get("test_name")
                    })
        
        except subprocess.TimeoutExpired:
            self.log("Bandit scan timeout", "ERROR")
        except json.JSONDecodeError as e:
            self.log(f"Failed to parse Bandit output: {e}", "ERROR")
        except Exception as e:
            self.log(f"Bandit scan failed: {e}", "ERROR")
        
        return {
            "tool": "bandit",
            "scan_time": datetime.now().isoformat(),
            "issues": issues,
            "total": len(issues)
        }
    
    def check_outdated_packages(self) -> Dict[str, Any]:
        """Check for outdated packages"""
        self.log("Checking for outdated packages...", "INFO")
        outdated = []
        
        try:
            # Check Python packages
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format", "json"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.stdout:
                data = json.loads(result.stdout)
                for pkg in data:
                    outdated.append({
                        "package": pkg.get("name"),
                        "current_version": pkg.get("version"),
                        "latest_version": pkg.get("latest_version"),
                        "type": "python"
                    })
        
        except Exception as e:
            self.log(f"Failed to check outdated packages: {e}", "ERROR")
        
        return {
            "scan_time": datetime.now().isoformat(),
            "outdated_packages": outdated,
            "total": len(outdated)
        }
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive security report"""
        report = []
        report.append("=" * 80)
        report.append("DEPENDENCY SECURITY SCAN REPORT")
        report.append("=" * 80)
        report.append(f"Scan Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary
        total_vulns = sum(r.get("total", 0) for r in results.values() if "vulnerabilities" in r or "issues" in r)
        report.append(f"SUMMARY: Found {total_vulns} security issues")
        report.append("-" * 40)
        
        # Critical vulnerabilities
        critical_count = 0
        for tool_name, result in results.items():
            if "vulnerabilities" in result:
                for vuln in result["vulnerabilities"]:
                    if vuln.get("severity", "").lower() in ["critical", "high"]:
                        critical_count += 1
        
        if critical_count > 0:
            report.append(f"⚠️  CRITICAL/HIGH SEVERITY: {critical_count} vulnerabilities")
            report.append("")
        
        # Detailed results by tool
        for tool_name, result in results.items():
            if result.get("total", 0) > 0:
                report.append(f"\n{tool_name.upper()} Results:")
                report.append("-" * 40)
                
                if "vulnerabilities" in result:
                    for vuln in result["vulnerabilities"]:
                        report.append(f"  • Package: {vuln.get('package')}")
                        report.append(f"    Severity: {vuln.get('severity', 'unknown')}")
                        report.append(f"    Description: {vuln.get('description', 'N/A')[:100]}...")
                        if vuln.get("fixed_version"):
                            report.append(f"    Fix: Update to {vuln['fixed_version']}")
                        report.append("")
                
                elif "issues" in result:
                    for issue in result["issues"][:10]:  # Limit to first 10
                        report.append(f"  • File: {issue.get('file')}")
                        report.append(f"    Line: {issue.get('line')}")
                        report.append(f"    Severity: {issue.get('severity')}")
                        report.append(f"    Issue: {issue.get('issue')}")
                        report.append("")
                
                elif "outdated_packages" in result:
                    for pkg in result["outdated_packages"][:20]:  # Limit to first 20
                        report.append(f"  • {pkg['package']}: {pkg['current_version']} → {pkg['latest_version']}")
        
        # Recommendations
        report.append("\nRECOMMENDATIONS:")
        report.append("-" * 40)
        
        if critical_count > 0:
            report.append("1. URGENT: Address critical/high severity vulnerabilities immediately")
        
        report.append("2. Update all outdated packages to latest versions")
        report.append("3. Review and fix code security issues identified by Bandit")
        report.append("4. Set up automated dependency scanning in CI/CD pipeline")
        report.append("5. Subscribe to security advisories for your dependencies")
        
        return "\n".join(report)
    
    def save_report(self, results: Dict[str, Any], format: str = "all"):
        """Save scan results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON report
        if format in ["json", "all"]:
            json_file = self.report_dir / f"security_scan_{timestamp}.json"
            with open(json_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            self.log(f"JSON report saved to: {json_file}", "INFO")
        
        # Save text report
        if format in ["text", "all"]:
            text_report = self.generate_report(results)
            text_file = self.report_dir / f"security_scan_{timestamp}.txt"
            text_file.write_text(text_report)
            self.log(f"Text report saved to: {text_file}", "INFO")
            
            # Also print to console
            print("\n" + text_report)
    
    def run_full_scan(self) -> Dict[str, Any]:
        """Run complete security scan"""
        self.log("Starting comprehensive dependency security scan...", "INFO")
        self.scan_timestamp = datetime.now()
        
        # Check tool availability
        tools = self.check_tool_availability()
        
        results = {}
        
        # Run available scans
        if tools.get("safety"):
            results["safety"] = self.scan_python_with_safety()
        
        if tools.get("pip-audit"):
            results["pip_audit"] = self.scan_python_with_pip_audit()
        
        if tools.get("npm"):
            results["npm_audit"] = self.scan_npm_dependencies()
        
        if tools.get("bandit"):
            results["bandit"] = self.scan_code_with_bandit()
        
        # Always check outdated packages
        results["outdated"] = self.check_outdated_packages()
        
        # Add metadata
        results["metadata"] = {
            "scan_timestamp": self.scan_timestamp.isoformat(),
            "tools_available": tools,
            "python_version": sys.version,
            "platform": sys.platform
        }
        
        return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Dependency Security Scanner")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-f", "--format", choices=["json", "text", "all"], default="all",
                      help="Output format (default: all)")
    parser.add_argument("--install-tools", action="store_true",
                      help="Install missing security tools")
    
    args = parser.parse_args()
    
    if args.install_tools:
        print("Installing security tools...")
        tools = [
            "pip install safety",
            "pip install pip-audit",
            "pip install bandit",
            "pip install semgrep"
        ]
        for cmd in tools:
            print(f"Running: {cmd}")
            subprocess.run(cmd.split(), check=False)
        print("Installation complete. Run the scanner again.")
        return
    
    scanner = DependencySecurityScanner(verbose=args.verbose)
    results = scanner.run_full_scan()
    scanner.save_report(results, format=args.format)
    
    # Exit with error code if vulnerabilities found
    total_issues = sum(r.get("total", 0) for r in results.values() if isinstance(r, dict))
    if total_issues > 0:
        sys.exit(1)
    else:
        print("\n✅ No security vulnerabilities found!")
        sys.exit(0)


if __name__ == "__main__":
    main()