# Local RAG System 🔍

A powerful Retrieval-Augmented Generation (RAG) system for querying PDF documents using LLMs. Built with LangChain, ChromaDB, and support for both OpenAI and Anthropic models.

## Features ✨

- 📄 **PDF Processing**: Automatic text extraction and intelligent chunking
- 🔍 **Semantic Search**: Vector-based document retrieval using ChromaDB
- 🤖 **Multiple LLM Support**: OpenAI GPT and Anthropic Claude integration
- 💬 **Interactive UI**: Web interface built with Gradio
- 🚀 **Local Embeddings**: Option to use Sentence Transformers for privacy
- 📊 **Source Attribution**: Always shows which documents were used for answers

## Demo

![RAG System Demo](docs/demo.gif)

## Quick Start 🚀

### Prerequisites

- Python 3.8+
- Git
- (Optional) CUDA-capable GPU for faster embeddings

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/local-rag-system.git
cd local-rag-system