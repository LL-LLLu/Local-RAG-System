import gradio as gr
from query import create_rag_chain, search_documents
from ingest import load_documents, split_documents, create_vectorstore
from config import *
import os
from pathlib import Path
import time

# Initialize the RAG chain globally
rag_chain = None

def initialize_rag():
    """Initialize the RAG chain"""
    global rag_chain
    try:
        rag_chain = create_rag_chain()
        return "✅ RAG system initialized successfully!"
    except Exception as e:
        return f"❌ Failed to initialize: {str(e)}"

def query_documents(question, num_sources=5):
    """Query the documents and return answer with sources"""
    if not rag_chain:
        return "⚠️ Please initialize the system first!", ""
    
    if not question.strip():
        return "⚠️ Please enter a question!", ""
    
    try:
        # Get response
        result = rag_chain({"query": question})
        
        # Format answer
        answer = result['result']
        
        # Format sources
        sources = "\n\n".join([
            f"📄 **Source {i+1}**: {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'N/A')})\n{doc.page_content[:200]}..."
            for i, doc in enumerate(result['source_documents'][:num_sources])
        ])
        
        return answer, sources
    
    except Exception as e:
        return f"❌ Error: {str(e)}", ""

def upload_and_process_pdfs(files):
    """Handle PDF uploads and processing"""
    if not files:
        return "⚠️ No files uploaded!"
    
    try:
        # Save uploaded files
        uploaded_files = []
        for file in files:
            filename = os.path.basename(file.name)
            filepath = DATA_DIR / filename
            
            # Copy file to data directory
            with open(file.name, 'rb') as src, open(filepath, 'wb') as dst:
                dst.write(src.read())
            uploaded_files.append(filename)
        
        # Load and process documents
        documents = load_documents()
        chunks = split_documents(documents)
        create_vectorstore(chunks)
        
        return f"✅ Successfully processed {len(uploaded_files)} files:\n" + "\n".join(uploaded_files) + f"\n\n📊 Created {len(chunks)} chunks from {len(documents)} pages"
    
    except Exception as e:
        return f"❌ Error processing files: {str(e)}"

def get_system_status():
    """Get current system status"""
    try:
        # Check PDFs
        pdf_files = list(DATA_DIR.glob("*.pdf"))
        pdf_count = len(pdf_files)
        
        # Check vector store
        vectorstore_exists = VECTORSTORE_DIR.exists() and any(VECTORSTORE_DIR.iterdir())
        
        # Get file list
        file_list = "\n".join([f"• {pdf.name}" for pdf in pdf_files[:10]])
        if pdf_count > 10:
            file_list += f"\n... and {pdf_count - 10} more"
        
        status = f"""
📊 **System Status**

**PDF Files:** {pdf_count} files
{file_list if pdf_count > 0 else "No files uploaded yet"}

**Vector Store:** {'✅ Ready' if vectorstore_exists else '❌ Not created'}
**LLM Provider:** {LLM_PROVIDER} ({LLM_MODEL})
**Embedding Model:** {EMBEDDING_MODEL}
**Chunk Size:** {CHUNK_SIZE} characters
"""
        return status
    except Exception as e:
        return f"❌ Error getting status: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Local RAG System", theme=gr.themes.Soft()) as app:
    gr.Markdown(
        """
        # 🔍 Local RAG System
        
        Query your PDF documents using AI. Upload PDFs, ask questions, and get answers with sources.
        """
    )
    
    with gr.Tab("💬 Query Documents"):
        with gr.Row():
            with gr.Column(scale=2):
                query_input = gr.Textbox(
                    label="Ask a question",
                    placeholder="What is the main topic of the documents?",
                    lines=2
                )
                num_sources_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Number of sources to show"
                )
                query_btn = gr.Button("🔍 Search", variant="primary")
                
            with gr.Column(scale=1):
                init_btn = gr.Button("🚀 Initialize System", variant="secondary")
                init_status = gr.Textbox(label="Status", lines=2)
        
        answer_output = gr.Markdown(label="Answer")
        sources_output = gr.Markdown(label="Sources")
        
        # Event handlers
        init_btn.click(
            fn=initialize_rag,
            outputs=init_status
        )
        
        query_btn.click(
            fn=query_documents,
            inputs=[query_input, num_sources_slider],
            outputs=[answer_output, sources_output]
        )
        
        query_input.submit(
            fn=query_documents,
            inputs=[query_input, num_sources_slider],
            outputs=[answer_output, sources_output]
        )
    
    with gr.Tab("📄 Upload Documents"):
        file_upload = gr.File(
            label="Upload PDF files",
            file_types=[".pdf"],
            file_count="multiple"
        )
        upload_btn = gr.Button("📤 Process PDFs", variant="primary")
        upload_status = gr.Textbox(label="Processing Status", lines=5)
        
        upload_btn.click(
            fn=upload_and_process_pdfs,
            inputs=file_upload,
            outputs=upload_status
        )
    
    with gr.Tab("⚙️ System Status"):
        status_display = gr.Markdown()
        refresh_btn = gr.Button("🔄 Refresh Status")
        
        refresh_btn.click(
            fn=get_system_status,
            outputs=status_display
        )
        
        # Load status on tab load
        app.load(fn=get_system_status, outputs=status_display)
    
    with gr.Tab("📚 Examples"):
        gr.Markdown(
            """
            ### Example Questions:
            
            - What is the main topic of these documents?
            - Can you summarize the key findings?
            - What methodology was used in the research?
            - What are the limitations mentioned?
            - Compare and contrast the different approaches discussed
            - What are the future research directions suggested?
            
            ### Tips:
            
            1. **Be specific**: Instead of "Tell me about AI", ask "What specific AI techniques are discussed?"
            2. **Ask for comparisons**: "How does approach A differ from approach B?"
            3. **Request summaries**: "Summarize the main contributions in bullet points"
            4. **Seek clarification**: "Explain the concept of X in simple terms"
            """
        )

# Launch the app
if __name__ == "__main__":
    print("Starting RAG Web Interface...")
    print("Make sure to:")
    print("1. Upload PDFs in the 'Upload Documents' tab")
    print("2. Click 'Initialize System' in the 'Query Documents' tab")
    print("3. Start asking questions!")
    
    app.launch(
        share=False,  # Set to True to create a public link
        server_port=7860,
        server_name="127.0.0.1"
    )