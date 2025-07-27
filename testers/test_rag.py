import os
import sys
from pathlib import Path
import time
from config import *
from ingest import load_documents, split_documents, create_vectorstore
from query import search_documents, create_rag_chain, load_vectorstore

def test_configuration():
    """Test 1: Check configuration and dependencies"""
    print("="*50)
    print("TEST 1: Configuration Check")
    print("="*50)
    
    # Check directories
    assert DATA_DIR.exists(), f"Data directory {DATA_DIR} does not exist"
    assert VECTORSTORE_DIR.exists(), f"Vector store directory {VECTORSTORE_DIR} does not exist"
    print("[OK] Directories exist")
    
    # Check for PDFs
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    assert len(pdf_files) > 0, "No PDF files found in data/pdfs directory"
    print(f"[OK] Found {len(pdf_files)} PDF files")
    
    # Check API keys if using online LLMs
    if LLM_PROVIDER == "openai":
        assert OPENAI_API_KEY, "OPENAI_API_KEY not set"
        print("[OK] OpenAI API key configured")
    elif LLM_PROVIDER == "anthropic":
        assert ANTHROPIC_API_KEY, "ANTHROPIC_API_KEY not set"
        print("[OK] Anthropic API key configured")
    
    print("\nConfiguration test passed!\n")

def test_document_loading():
    """Test 2: Document loading and processing"""
    print("="*50)
    print("TEST 2: Document Loading")
    print("="*50)
    
    # Load documents
    documents = load_documents()
    assert len(documents) > 0, "No documents loaded"
    print(f"[OK] Loaded {len(documents)} document pages")
    
    # Check document structure
    first_doc = documents[0]
    assert hasattr(first_doc, 'page_content'), "Document missing page_content"
    assert hasattr(first_doc, 'metadata'), "Document missing metadata"
    assert 'source' in first_doc.metadata, "Document missing source in metadata"
    print("[OK] Document structure is correct")
    
    # Split documents
    chunks = split_documents(documents)
    assert len(chunks) >= len(documents), "Chunking failed"
    print(f"[OK] Split into {len(chunks)} chunks")
    
    # Check chunk size
    avg_chunk_size = sum(len(chunk.page_content) for chunk in chunks) / len(chunks)
    print(f"[OK] Average chunk size: {avg_chunk_size:.0f} characters")
    
    print("\nDocument loading test passed!\n")
    return chunks

def test_vector_store(chunks):
    """Test 3: Vector store creation and retrieval"""
    print("="*50)
    print("TEST 3: Vector Store")
    print("="*50)
    
    # Create vector store
    print("Creating vector store...")
    start_time = time.time()
    vectorstore = create_vectorstore(chunks)
    creation_time = time.time() - start_time
    print(f"[OK] Vector store created in {creation_time:.2f} seconds")
    
    # Test loading vector store
    loaded_vectorstore = load_vectorstore()
    print("[OK] Vector store loaded successfully")
    
    # Test similarity search
    test_queries = [
        "What is the main topic of this document?",
        "machine learning",
        "artificial intelligence",
        "test query that probably won't match anything specific"
    ]
    
    for query in test_queries:
        results = search_documents(query, k=3)
        print(f"\n[OK] Query: '{query}' returned {len(results)} results")
        if results:
            print(f"  Top result score: {results[0][1]:.4f}")
    
    print("\nVector store test passed!\n")

def test_llm_integration():
    """Test 4: LLM integration and answer generation"""
    print("="*50)
    print("TEST 4: LLM Integration")
    print("="*50)
    
    try:
        # Create RAG chain
        qa_chain = create_rag_chain()
        print("[OK] RAG chain created successfully")
        
        # Test queries
        test_questions = [
            "What is this document about?",
            "Can you summarize the main points?",
            "What are the key findings or conclusions?"
        ]
        
        for question in test_questions:
            print(f"\nTesting question: '{question}'")
            start_time = time.time()
            
            result = qa_chain({"query": question})
            response_time = time.time() - start_time
            
            assert 'result' in result, "No result in response"
            assert 'source_documents' in result, "No source documents in response"
            assert len(result['result']) > 0, "Empty response"
            
            print(f"[OK] Got response in {response_time:.2f} seconds")
            print(f"  Response length: {len(result['result'])} characters")
            print(f"  Sources used: {len(result['source_documents'])}")
            print(f"  Preview: {result['result'][:100]}...")
        
        print("\nLLM integration test passed!\n")
        
    except Exception as e:
        print(f"[FAIL] LLM integration failed: {e}")
        print("Check your API keys and network connection")
        return False
    
    return True

def test_edge_cases():
    """Test 5: Edge cases and error handling"""
    print("="*50)
    print("TEST 5: Edge Cases")
    print("="*50)
    
    # Test empty query
    try:
        results = search_documents("", k=5)
        print("[OK] Empty query handled")
    except Exception as e:
        print(f"[FAIL] Empty query failed: {e}")
    
    # Test very long query
    long_query = "test " * 100
    try:
        results = search_documents(long_query, k=5)
        print("[OK] Long query handled")
    except Exception as e:
        print(f"[FAIL] Long query failed: {e}")
    
    # Test special characters
    special_query = "test!@#$%^&*()_+-=[]{}|;':\",./<>?"
    try:
        results = search_documents(special_query, k=5)
        print("[OK] Special characters handled")
    except Exception as e:
        print(f"[FAIL] Special characters failed: {e}")
    
    print("\nEdge case test completed!\n")

def run_performance_test():
    """Test 6: Performance benchmarks"""
    print("="*50)
    print("TEST 6: Performance Benchmarks")
    print("="*50)
    
    # Benchmark search speed
    queries = ["machine learning", "artificial intelligence", "data science", "neural networks", "deep learning"]
    
    search_times = []
    for query in queries:
        start_time = time.time()
        results = search_documents(query, k=5)
        search_time = time.time() - start_time
        search_times.append(search_time)
    
    avg_search_time = sum(search_times) / len(search_times)
    print(f"[OK] Average search time: {avg_search_time:.3f} seconds")
    
    # Benchmark RAG response time (if using LLM)
    if LLM_PROVIDER in ["openai", "anthropic"]:
        try:
            qa_chain = create_rag_chain()
            
            response_times = []
            for query in queries[:3]:  # Only test 3 to save API costs
                start_time = time.time()
                result = qa_chain({"query": f"What can you tell me about {query}?"})
                response_time = time.time() - start_time
                response_times.append(response_time)
            
            avg_response_time = sum(response_times) / len(response_times)
            print(f"[OK] Average RAG response time: {avg_response_time:.2f} seconds")
        except:
            print("[WARNING] Skipping LLM performance test")
    
    print("\nPerformance test completed!\n")

def main():
    """Run all tests"""
    print("\n*** RAG SYSTEM TEST SUITE ***\n")
    
    # Run tests in sequence
    try:
        # Test 1: Configuration
        test_configuration()
        
        # Test 2: Document Loading
        chunks = test_document_loading()
        
        # Test 3: Vector Store
        test_vector_store(chunks[:10])  # Use only first 10 chunks for speed
        
        # Test 4: LLM Integration (optional)
        if input("Run LLM integration test? This will use API credits (y/n): ").lower() == 'y':
            test_llm_integration()
        
        # Test 5: Edge Cases
        test_edge_cases()
        
        # Test 6: Performance
        if input("Run performance benchmarks? (y/n): ").lower() == 'y':
            run_performance_test()
        
        print("\n*** ALL TESTS COMPLETED ***\n")
        
    except AssertionError as e:
        print(f"\n[ERROR] TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] UNEXPECTED ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()