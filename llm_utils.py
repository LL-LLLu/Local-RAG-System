try:
    # Try newer import style first
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
except ImportError:
    # Fall back to older import style
    from langchain_openai import ChatOpenAI, ChatAnthropic

from langchain.schema import HumanMessage, SystemMessage
from config import *

# from LocalRAG.config import ANTHROPIC_API_KEY, LLM_MODEL, LLM_PROVIDER, LLM_TEMPERATURE, MAX_TOKENS, OPENAI_API_KEY
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage
from config import *


def get_llm():
    """Initialize and return the appropriate LLM"""
    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in .env file")
        return ChatOpenAI(
            model_name=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=MAX_TOKENS,
            openai_api_key=OPENAI_API_KEY
        )
    elif LLM_PROVIDER == "anthropic":
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not set in .env file")
        return ChatAnthropic(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=MAX_TOKENS,
            anthropic_api_key=ANTHROPIC_API_KEY
        )
    else:
        raise ValueError(f"Unknown LLM provider: {LLM_PROVIDER}")

def create_prompt(question, context_docs):
    """Create a prompt for the LLM with context"""
    context = "\n\n".join([
        f"[Document {i+1}]\n{doc.page_content}"
        for i, doc in enumerate(context_docs)
    ])
    
    prompt = f"""You are a helpful assistant answering questions based on the provided documents.
Use only the information from the documents below to answer the question.
If the answer cannot be found in the documents, say so clearly.

Documents:
{context}

Question: {question}

Answer:"""
    
    return prompt