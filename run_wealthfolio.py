#!/usr/bin/env python3
"""
Wealthfolio Portfolio Strategy System - Wrapper Script

This script ensures the correct Python version is used and provides helpful error messages
when dependencies are missing.
"""

import sys
import subprocess
import os

def check_dependencies():
    """Check if required dependencies are available."""
    missing_deps = []
    
    try:
        import google.generativeai
    except ImportError:
        missing_deps.append("google-generativeai")
    
    try:
        import ddgs
    except ImportError:
        missing_deps.append("ddgs")
    
    try:
        import trafilatura
    except ImportError:
        missing_deps.append("trafilatura")
    
    return missing_deps

def main():
    """Main entry point that ensures correct Python version and dependencies."""
    
    # Check if we're running with python3
    if sys.version_info[0] < 3:
        print("❌ Error: Python 3 is required to run Wealthfolio Portfolio Strategy System")
        print("💡 Please use: python3 src/main.py [command] [options]")
        sys.exit(1)
    
    # Check dependencies
    missing_deps = check_dependencies()
    
    if missing_deps:
        print("❌ Missing required dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\n💡 To install missing dependencies, run:")
        print("   pip3 install -r requirements.txt")
        print("\n💡 Then run:")
        print("   python3 src/main.py [command] [options]")
        sys.exit(1)
    
    # Import and run the main module
    try:
        # Change to the script's directory to ensure relative paths work
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        # Import the main module
        from src.main import main as wealthfolio_main
        wealthfolio_main()
        
    except ImportError as e:
        print(f"❌ Error importing main module: {e}")
        print("💡 Make sure you're running this script from the project root directory")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()