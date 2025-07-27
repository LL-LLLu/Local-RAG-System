# hybrid_search.py
import numpy as np  # This is what 'np' stands for
from rank_bm25 import BM25Okapi
from typing import List, Tuple, Dict
from langchain.schema import Document

# Install required packages:
# pip install numpy rank-bm25

class HybridRetriever:
    def __init__(self, vectorstore, documents):
        self.vectorstore = vectorstore
        self.documents = documents
        
        # Prepare BM25
        tokenized_docs = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
    def search(self, query: str, k: int = 5, alpha: float = 0.5) -> List[Tuple[Document, float]]:
        """
        Hybrid search combining vector and BM25 search
        
        Args:
            query: Search query
            k: Number of results to return
            alpha: Weight for vector search (0-1). Higher = more weight on vector search
        
        Returns:
            List of (document, score) tuples
        """
        # Vector search
        vector_results = self.vectorstore.similarity_search_with_score(query, k=k*2)
        
        # BM25 search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[-k*2:][::-1]  # np.argsort sorts and returns indices
        
        # Combine scores
        combined_results = {}
        
        # Add vector search results
        for doc, score in vector_results:
            doc_id = doc.metadata.get('doc_id', str(doc))
            combined_results[doc_id] = {
                'doc': doc,
                'vector_score': score,
                'bm25_score': 0,
                'combined_score': score * alpha
            }
        
        # Add BM25 results
        for idx in bm25_top_indices:
            doc = self.documents[idx]
            doc_id = doc.metadata.get('doc_id', str(doc))
            if doc_id in combined_results:
                combined_results[doc_id]['bm25_score'] = bm25_scores[idx]
                combined_results[doc_id]['combined_score'] += bm25_scores[idx] * (1 - alpha)
            else:
                combined_results[doc_id] = {
                    'doc': doc,
                    'vector_score': 0,
                    'bm25_score': bm25_scores[idx],
                    'combined_score': bm25_scores[idx] * (1 - alpha)
                }
        
        # Sort by combined score
        sorted_results = sorted(
            combined_results.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )[:k]
        
        return [(r['doc'], r['combined_score']) for r in sorted_results]