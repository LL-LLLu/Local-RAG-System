from query import create_rag_chain
import time

# Initialize
qa_chain = create_rag_chain()

test_cases = [
    # (query, expected_behavior)
    ("What is the main topic of these documents?", "Should return general overview"),
    ("What specific algorithms are mentioned?", "Should return technical details"),
    ("Who are the authors?", "Should return author information"),
    ("What year was this published?", "Should return publication date"),
    ("Compare transformer and RNN architectures", "Should provide comparison"),
    ("What are the limitations?", "Should discuss limitations"),
    ("How many parameters does the model have?", "Should return specific numbers"),
    ("What datasets were used?", "Should list datasets"),
    ("Random gibberish asdkfj", "Should handle gracefully"),
    ("", "Should handle empty query"),
]

for query, expected in test_cases:
    print(f"\nQuery: {query}")
    print(f"Expected: {expected}")
    
    try:
        start = time.time()
        result = qa_chain({"query": query})
        elapsed = time.time() - start
        
        print(f"✓ Responded in {elapsed:.2f}s")
        print(f"Answer length: {len(result['result'])} chars")
        print(f"Sources: {len(result['source_documents'])}")
        
        # Show preview
        preview = result['result'][:150] + "..." if len(result['result']) > 150 else result['result']
        print(f"Preview: {preview}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("-" * 50)