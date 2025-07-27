import time
from query import create_rag_chain, search_documents
import statistics

print("Performance Testing")
print("=" * 50)

# Test 1: Vector search performance
print("\n1. Vector Search Performance")
queries = ["AI", "machine learning", "transformer", "attention", "neural network"]
times = []

for q in queries:
    start = time.time()
    results = search_documents(q, k=5)
    elapsed = time.time() - start
    times.append(elapsed)
    print(f"Query '{q}': {elapsed:.3f}s ({len(results)} results)")

print(f"\nAverage search time: {statistics.mean(times):.3f}s")
print(f"Min/Max: {min(times):.3f}s / {max(times):.3f}s")

# Test 2: RAG performance
print("\n2. RAG Response Performance")
qa_chain = create_rag_chain()
rag_times = []

for q in queries[:3]:  # Only test 3 to save API costs
    start = time.time()
    result = qa_chain({"query": f"Explain {q} in simple terms"})
    elapsed = time.time() - start
    rag_times.append(elapsed)
    print(f"Query '{q}': {elapsed:.2f}s")

print(f"\nAverage RAG time: {statistics.mean(rag_times):.2f}s")