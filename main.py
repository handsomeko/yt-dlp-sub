#!/usr/bin/env python3
"""
YouTube Content Intelligence & Repurposing Platform - Main Entry Point
Phase 1: CLI Tool
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Run startup validation before anything else
from core.startup_validation import run_startup_validation

# Validate at startup - exit if validation fails
print("🚨 Running startup validation...")
if not run_startup_validation(exit_on_error=True):
    print("❌ Startup validation failed. Please fix the issues and restart.")
    sys.exit(1)
print("✅ Startup validation passed")

from cli import cli


def main():
    """Main entry point for the application"""
    try:
        # Run the CLI
        cli()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()