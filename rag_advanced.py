# For newer versions of LangChain
try:
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
except ImportError:
    from langchain_openai import ChatOpenAI, ChatAnthropic


# from LocalRAG.config import ANTHROPIC_API_KEY, LLM_MODEL, LLM_PROVIDER, LLM_TEMPERATURE, MAX_TOKENS, OPENAI_API_KEY
# For newer versions of LangChain (0.1.0+)
from LocalRAG.llm_utils import get_llm
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from query import load_vectorstore
from config import *

def get_llm_for_advanced():
    """Get LLM with proper imports for newer LangChain"""
    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in .env file")
        return ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=MAX_TOKENS,
            api_key=OPENAI_API_KEY
        )
    elif LLM_PROVIDER == "anthropic":
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not set in .env file")
        return ChatAnthropic(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=MAX_TOKENS,
            api_key=ANTHROPIC_API_KEY
        )

def create_conversational_rag():
    """Create a RAG chain with conversation memory"""
    vectorstore = load_vectorstore()
    llm = get_llm()
    
    # Add conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    # Create conversational chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        return_source_documents=True,
        verbose=True
    )
    
    return qa_chain

# Example with streaming (for OpenAI)
def create_streaming_rag():
    """Create a RAG chain with streaming responses"""
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    
    vectorstore = load_vectorstore()
    
    # Create LLM with streaming
    llm = ChatOpenAI(
        model_name=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    
    return qa_chain