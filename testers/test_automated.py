import sys
import os
from pathlib import Path
import time
from config import *
from ingest import load_documents, split_documents, create_vectorstore
from query import search_documents, load_vectorstore

def test_configuration():
    """Test 1: Check configuration and dependencies"""
    print("="*50)
    print("TEST 1: Configuration Check")
    print("="*50)
    
    try:
        # Check directories
        assert DATA_DIR.exists(), f"Data directory {DATA_DIR} does not exist"
        assert VECTORSTORE_DIR.exists(), f"Vector store directory {VECTORSTORE_DIR} does not exist"
        print("[OK] Directories exist")
        
        # Check for PDFs
        pdf_files = list(DATA_DIR.glob("*.pdf"))
        if len(pdf_files) == 0:
            print("[WARNING] No PDF files found in data/pdfs directory")
            print("Please add some PDF files before running tests")
            return False
        print(f"[OK] Found {len(pdf_files)} PDF files")
        
        # Check API keys if using online LLMs
        if LLM_PROVIDER == "openai":
            if not OPENAI_API_KEY:
                print("[WARNING] OPENAI_API_KEY not set")
                return False
            print("[OK] OpenAI API key configured")
        elif LLM_PROVIDER == "anthropic":
            if not ANTHROPIC_API_KEY:
                print("[WARNING] ANTHROPIC_API_KEY not set")
                return False
            print("[OK] Anthropic API key configured")
        
        print("\nConfiguration test passed!\n")
        return True
    except Exception as e:
        print(f"[ERROR] Configuration test failed: {e}")
        return False

def test_basic_functionality():
    """Test 2: Basic functionality without LLM"""
    print("="*50)
    print("TEST 2: Basic Functionality")
    print("="*50)
    
    try:
        # Test document loading
        print("Testing document loading...")
        documents = load_documents()
        if len(documents) == 0:
            print("[FAIL] No documents loaded")
            return False
        print(f"[OK] Loaded {len(documents)} document pages")
        
        # Test chunking
        print("\nTesting document chunking...")
        chunks = split_documents(documents[:5])  # Test with first 5 pages only
        print(f"[OK] Split into {len(chunks)} chunks")
        
        # Test vector store
        print("\nTesting vector store operations...")
        if Path(VECTORSTORE_DIR).exists() and len(list(Path(VECTORSTORE_DIR).iterdir())) > 0:
            print("[OK] Vector store already exists, testing search...")
            
            # Test search
            test_query = "test query"
            results = search_documents(test_query, k=3)
            print(f"[OK] Search returned {len(results)} results")
        else:
            print("[INFO] Vector store doesn't exist, skipping search test")
            print("Run 'python ingest.py' to create vector store")
        
        print("\nBasic functionality test passed!\n")
        return True
        
    except Exception as e:
        print(f"[ERROR] Basic functionality test failed: {e}")
        return False

def test_imports():
    """Test all imports work correctly"""
    print("="*50)
    print("TEST 3: Import Test")
    print("="*50)
    
    modules_to_test = [
        ("ingest", "Document ingestion module"),
        ("query", "Query module"),
        ("config", "Configuration module"),
        ("llm_utils", "LLM utilities module")
    ]
    
    all_passed = True
    for module_name, description in modules_to_test:
        try:
            __import__(module_name)
            print(f"[OK] {description} imported successfully")
        except ImportError as e:
            print(f"[FAIL] Failed to import {module_name}: {e}")
            all_passed = False
    
    if all_passed:
        print("\nAll imports successful!\n")
    return all_passed

def main():
    """Run automated tests"""
    print("\n*** AUTOMATED RAG SYSTEM TESTS ***\n")
    print("This will run non-interactive tests only.\n")
    
    # Track results
    results = []
    
    # Test 1: Imports
    print("Running import tests...")
    results.append(("Import Test", test_imports()))
    
    # Test 2: Configuration
    print("\nRunning configuration tests...")
    results.append(("Configuration Test", test_configuration()))
    
    # Test 3: Basic functionality
    print("\nRunning basic functionality tests...")
    results.append(("Basic Functionality", test_basic_functionality()))
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n*** ALL AUTOMATED TESTS PASSED ***\n")
        print("Next steps:")
        print("1. Run 'python ingest.py' to ingest documents")
        print("2. Run 'python query.py' to test queries")
        print("3. Run 'python test_rag.py' for full interactive tests")
    else:
        print("\n*** SOME TESTS FAILED ***")
        print("Please fix the issues before proceeding.")
        sys.exit(1)

if __name__ == "__main__":
    main()