import os
import re

def fix_file(filepath):
    """Fix deprecated imports in a Python file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Define replacements
    replacements = [
        (r'from langchain\.document_loaders import PyPDFLoader',
         'from langchain_community.document_loaders import PyPDFLoader'),
        (r'from langchain\.embeddings import HuggingFaceEmbeddings',
         'from langchain_community.embeddings import HuggingFaceEmbeddings'),
        (r'from langchain\.vectorstores import Chroma',
         'from langchain_community.vectorstores import Chroma'),
        (r'from langchain\.chat_models import ChatOpenAI',
         'from langchain_openai import ChatOpenAI'),
        (r'from langchain\.chat_models import ChatAnthropic',
         'from langchain_anthropic import ChatAnthropic'),
    ]
    
    # Apply replacements
    for old, new in replacements:
        content = re.sub(old, new, content)
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed imports in {filepath}")

# Fix all Python files
for file in ['ingest.py', 'query.py', 'llm_utils.py', 'rag_advanced.py']:
    if os.path.exists(file):
        fix_file(file)

print("\nNow install missing package:")
print("pip install pypdf")