# memory_rag.py
from LocalRAG.query import *
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain

def create_conversational_rag():
    """Create RAG with conversation memory"""
    vectorstore = load_vectorstore()
    llm = get_llm()
    
    memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 5}
        ),
        memory=memory,
        return_source_documents=True,
        verbose=True
    )
    
    return qa_chain