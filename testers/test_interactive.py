from query import create_rag_chain, search_documents
from config import *

def interactive_test():
    """Interactive testing with predefined scenarios"""
    print("*** INTERACTIVE RAG TESTER ***\n")
    
    scenarios = {
        "1": {
            "name": "Basic Search Test",
            "queries": [
                "What is the main topic?",
                "Can you explain the key concepts?",
                "What are the conclusions?"
            ]
        },
        "2": {
            "name": "Specific Information Retrieval",
            "queries": [
                "What methodology was used?",
                "What are the limitations mentioned?",
                "Who are the authors?"
            ]
        },
        "3": {
            "name": "Complex Questions",
            "queries": [
                "Compare and contrast the different approaches mentioned",
                "What are the implications of these findings?",
                "How does this relate to previous work?"
            ]
        },
        "4": {
            "name": "Custom Queries",
            "queries": []
        }
    }
    
    # Initialize RAG chain
    print("Initializing RAG system...")
    try:
        qa_chain = create_rag_chain()
        print("[OK] System ready!\n")
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return
    
    while True:
        print("\nSelect a test scenario:")
        for key, scenario in scenarios.items():
            print(f"{key}. {scenario['name']}")
        print("q. Quit")
        
        choice = input("\nYour choice: ").strip()
        
        if choice.lower() == 'q':
            break
        
        if choice not in scenarios:
            print("Invalid choice!")
            continue
        
        if choice == "4":
            # Custom queries
            while True:
                query = input("\nEnter your question (or 'back' to return): ").strip()
                if query.lower() == 'back':
                    break
                
                print("\n[Searching...]")
                result = qa_chain({"query": query})
                
                print("\n[Answer]")
                print("-" * 50)
                print(result['result'])
                print("-" * 50)
                
                if input("\nShow sources? (y/n): ").lower() == 'y':
                    for i, doc in enumerate(result['source_documents'], 1):
                        print(f"\nSource {i}: {doc.metadata.get('source')}")
                        print(f"Page: {doc.metadata.get('page')}")
        else:
            # Predefined scenarios
            queries = scenarios[choice]["queries"]
            for query in queries:
                print(f"\n[Question] {query}")
                input("Press Enter to see answer...")
                
                result = qa_chain({"query": query})
                
                print("\n[Answer]")
                print("-" * 50)
                print(result['result'])
                print("-" * 50)
                
                input("\nPress Enter for next question...")

if __name__ == "__main__":
    interactive_test()