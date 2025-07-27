try:
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
import chromadb
from config import *
from llm_utils import get_llm, create_prompt
# query.py (updated section)
from hybrid_search import HybridRetriever
import pickle
from pathlib import Path
from pathlib import Path
from cache import QueryCache
from logger_config import setup_logger


# Initialize logger and cache
logger = setup_logger('rag_query')
query_cache = QueryCache()

def search_documents(query, k=5, use_cache=True):
    """Search for relevant documents with caching"""
    logger.info(f"Searching for: {query}")
    
    # Check cache first
    if use_cache:
        cached_result = query_cache.get(query, k)
        if cached_result:
            logger.info("Returning cached result")
            return cached_result
    
    try:
        # Perform actual search
        vectorstore = load_vectorstore()
        results = vectorstore.similarity_search_with_score(query, k=k)
        
        # Cache the results
        if use_cache:
            query_cache.set(query, results, k)
            logger.info("Cached search results")
        
        return results
        
    except Exception as e:
        logger.error(f"Search failed for query '{query}': {e}")
        raise

def load_embeddings():
    """Load the appropriate embeddings model"""
    if USE_OPENAI_EMBEDDINGS:
        return OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY
        )
    else:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )

def load_vectorstore():
    """Load the existing vector store"""
    embeddings = load_embeddings()
    
    vectorstore = Chroma(
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    
    return vectorstore

def search_documents(query, k=5):
    """Search for relevant documents"""
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search_with_score(query, k=k)
    return results

def generate_answer(question, context_docs):
    """Generate an answer using the LLM"""
    llm = get_llm()
    prompt = create_prompt(question, context_docs)
    
    response = llm.invoke(prompt).content  # Updated for newer LangChain
    return response

def create_rag_chain():

    logger.info("Creating RAG chain")
    
    try:
        vectorstore = load_vectorstore()
        llm = get_llm()
        """Create a full RAG chain using LangChain"""
    
        vectorstore = load_vectorstore()
        llm = get_llm()
        
        # Create custom prompt template
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Always mention which document(s) you're basing your answer on.

        Context:
        {context}

        Question: {question}

        Answer:"""
        
        PROMPT = PromptTemplate(
                template=prompt_template, 
                input_variables=["context", "question"]
            )
            
        # Create the RAG chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

        logger.info("RAG chain created successfully")
        return qa_chain
            
    except Exception as e:
        logger.error(f"Failed to create RAG chain: {e}")
        raise
    

def pretty_print_answer(result):
    """Display the answer and sources nicely"""
    print("\n" + "="*80)
    print("ANSWER:")
    print("="*80)
    print(result['result'])
    
    print("\n" + "="*80)
    print("SOURCES:")
    print("="*80)
    
    for i, doc in enumerate(result['source_documents'], 1):
        print(f"\n[Source {i}]")
        print(f"File: {doc.metadata.get('source', 'Unknown')}")
        print(f"Page: {doc.metadata.get('page', 'Unknown')}")
        print(f"Preview: {doc.page_content[:200]}...")
        print("-" * 40)

def main():
    print(f"PDF RAG System (Using {LLM_PROVIDER} - {LLM_MODEL})")
    print("Type 'quit' to exit\n")
    
    # Initialize the RAG chain
    try:
        qa_chain = create_rag_chain()
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        print("Please check your API keys in the .env file")
        return
    
    while True:
        query = input("\nEnter your question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        print(f"\nSearching and generating answer...")
        
        try:
            # Using the full RAG chain
            result = qa_chain({"query": query})
            pretty_print_answer(result)
            
        except Exception as e:
            print(f"Error generating answer: {e}")


def create_hybrid_retriever():
    """Create hybrid retriever with cached documents"""
    vectorstore = load_vectorstore()
    
    # Load all documents (you might want to cache this)
    docs_cache_path = Path("cache/documents.pkl")
    
    if docs_cache_path.exists():
        # Load cached documents
        with open(docs_cache_path, 'rb') as f:
            documents = pickle.load(f)
    else:
        # Recreate documents from vectorstore
        # This is a workaround - better to save documents during ingestion
        documents = []
        # You'll need to implement this based on your vectorstore
        
    hybrid_retriever = HybridRetriever(vectorstore, documents)
    return hybrid_retriever

def search_documents_hybrid(query, k=5, alpha=0.5):
    """Search using hybrid retrieval"""
    retriever = create_hybrid_retriever()
    results = retriever.search(query, k=k, alpha=alpha)
    return results

if __name__ == "__main__":
    main()