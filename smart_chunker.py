# smart_chunker.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
import re

class SmartChunker:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def chunk_by_sections(self, text):
        """Split by document sections (headers, paragraphs)"""
        # Detect headers
        header_pattern = r'^#{1,6}\s+.+$|^.+\n[=-]+$'
        
        sections = []
        current_section = []
        current_header = None
        
        for line in text.split('\n'):
            if re.match(header_pattern, line, re.MULTILINE):
                # Save previous section
                if current_section:
                    sections.append({
                        'header': current_header,
                        'content': '\n'.join(current_section)
                    })
                current_header = line
                current_section = []
            else:
                current_section.append(line)
        
        # Don't forget the last section
        if current_section:
            sections.append({
                'header': current_header,
                'content': '\n'.join(current_section)
            })
        
        return sections
    
    def chunk_with_context(self, documents):
        """Add context to each chunk"""
        chunks = []
        
        for doc in documents:
            # Get document title/filename
            doc_title = doc.metadata.get('source', 'Unknown')
            
            # Split into sections
            sections = self.chunk_by_sections(doc.page_content)
            
            for section in sections:
                # Add context
                context = f"Document: {doc_title}\n"
                if section['header']:
                    context += f"Section: {section['header']}\n"
                
                # Create chunk with context
                chunk_content = context + "\n" + section['content']
                
                # Further split if too long
                if len(chunk_content) > self.chunk_size:
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap
                    )
                    sub_chunks = splitter.split_text(chunk_content)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(chunk_content)
        
        return chunks