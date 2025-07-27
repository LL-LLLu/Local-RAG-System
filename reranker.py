# reranker.py
from sentence_transformers import CrossEncoder
from numpy import np

class Reranker:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query, documents, top_k=5):
        # Prepare pairs
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Get scores
        scores = self.model.predict(pairs)
        
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1][:top_k]
        
        return [documents[i] for i in sorted_indices]