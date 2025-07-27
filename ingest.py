import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
# ingest.py (add this to your existing code)
import pickle
from config import *
# ingest.py (updated section)
from multimodal import MultiModalProcessor
from langchain.schema import Document


def load_documents_multimodal():
    """Load all PDFs including images and tables"""
    documents = []
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    
    # Initialize multimodal processor
    # For Windows, you might need to specify tesseract path:
    # processor = MultiModalProcessor(tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe')
    processor = MultiModalProcessor()
    
    for pdf_path in pdf_files:
        print(f"Processing {pdf_path.name}...")
        
        # Regular text extraction (existing code)
        loader = PyPDFLoader(str(pdf_path))
        text_docs = loader.load()
        documents.extend(text_docs)
        
        # Extract images and tables
        multimodal_data = processor.process_pdf_complete(str(pdf_path))
        
        # Add image OCR text as documents
        for img_data in multimodal_data['images']:
            doc = Document(
                page_content=f"[Image from page {img_data['page']}]\n{img_data['text']}",
                metadata={
                    'source': img_data['source'],
                    'page': img_data['page'],
                    'type': 'image_ocr'
                }
            )
            documents.append(doc)
        
        # Add tables as documents
        for table_data in multimodal_data['tables']:
            doc = Document(
                page_content=f"[Table from page {table_data['page']}]\n{table_data['text']}",
                metadata={
                    'source': table_data['source'],
                    'page': table_data['page'],
                    'type': 'table'
                }
            )
            documents.append(doc)
    
    print(f"Loaded {len(documents)} total documents (including images and tables)")
    return documents



def load_documents():
    """Load all PDFs from the data directory"""
    documents = []
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {DATA_DIR}")
        return documents
    
    for pdf_path in pdf_files:
        print(f"Loading {pdf_path.name}...")
        loader = PyPDFLoader(str(pdf_path))
        documents.extend(loader.load())
    
    print(f"Loaded {len(documents)} pages from {len(pdf_files)} PDFs")
    return documents

def split_documents(documents):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks

def create_vectorstore(chunks):
    """Create and populate the vector store"""
    # Initialize embeddings (this will download the model on first run)
    print("Initializing embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    
    # Create or update the vector store
    print("Creating vector store...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
        collection_name=COLLECTION_NAME
    )

        # Save documents for hybrid search
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    
    with open(cache_dir / "documents.pkl", 'wb') as f:
        pickle.dump(chunks, f)
    
    # Also save with doc_id for easier retrieval
    for i, chunk in enumerate(chunks):
        chunk.metadata['doc_id'] = f"doc_{i}"
    
    print(f"Vector store created with {len(chunks)} chunks")
    return vectorstore

def main():
    print("Starting document ingestion...")
    
    # Load documents
    documents = load_documents()
    if not documents:
        return
    
    # Split into chunks
    chunks = split_documents(documents)
    
    # Create vector store
    vectorstore = create_vectorstore(chunks)
    
    print("\nIngestion complete!")
    print(f"Vector store saved to: {VECTORSTORE_DIR}")

if __name__ == "__main__":
    main()