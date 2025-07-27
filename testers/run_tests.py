import sys
import os
import subprocess
import shutil
from pathlib import Path

def run_command(cmd, interactive=False):
    """Run a command and print output"""
    print(f"Running: {cmd}")
    
    if interactive:
        # For interactive commands, don't capture output
        result = subprocess.run(cmd, shell=True)
        return result.returncode == 0
    else:
        # For non-interactive commands, capture output
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"Error: {result.stderr}")
        return result.returncode == 0

def setup():
    """Install dependencies and create directories"""
    print("[SETUP] Setting up environment...")
    
    # Install requirements
    if not run_command("pip install -r requirements.txt"):
        print("Failed to install requirements!")
        return False
    
    # Create directories
    Path("data/pdfs").mkdir(parents=True, exist_ok=True)
    Path("vectorstore").mkdir(parents=True, exist_ok=True)
    
    print("[OK] Setup complete!")
    return True

def quick_test():
    """Quick import test"""
    print("[TEST] Running quick tests...")
    
    tests = [
        ("from ingest import *", "Ingest imports"),
        ("from query import *", "Query imports"),
        ("from config import *; print(f'Using {LLM_PROVIDER} - {LLM_MODEL}')", "Config check")
    ]
    
    for code, name in tests:
        if run_command(f'python -c "{code}"'):
            print(f"[OK] {name} working")
        else:
            print(f"[FAIL] {name} failed")
            return False
    
    return True

def full_test():
    """Run full test suite"""
    print("[TEST] Running full test suite...")
    print("Note: This test is interactive and will ask questions.")
    return run_command("python test_rag.py", interactive=True)

def interactive_test():
    """Run interactive tests"""
    print("[TEST] Starting interactive test...")
    return run_command("python test_interactive.py", interactive=True)

def ingest():
    """Run document ingestion"""
    print("[INGEST] Starting document ingestion...")
    return run_command("python ingest.py")

def query():
    """Run query interface"""
    print("[QUERY] Starting query interface...")
    return run_command("python query.py", interactive=True)

def clean():
    """Clean up vectorstore and cache"""
    print("[CLEAN] Cleaning up...")
    
    # Remove vectorstore
    if Path("vectorstore").exists():
        shutil.rmtree("vectorstore")
        Path("vectorstore").mkdir()
    
    # Remove cache
    if Path("__pycache__").exists():
        shutil.rmtree("__pycache__")
    
    print("[OK] Cleanup complete!")
    return True

def main():
    commands = {
        "setup": setup,
        "quick": quick_test,
        "full": full_test,
        "interactive": interactive_test,
        "ingest": ingest,
        "query": query,
        "clean": clean
    }
    
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print("RAG System Test Runner")
        print("\nUsage: python run_tests.py [command]")
        print("\nCommands:")
        for cmd in commands:
            print(f"  {cmd}")
        return
    
    command = sys.argv[1]
    success = commands[command]()
    
    if success:
        print(f"\n[OK] {command} completed successfully!")
    else:
        print(f"\n[FAIL] {command} failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()