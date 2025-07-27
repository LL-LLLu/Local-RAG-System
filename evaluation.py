# evaluation.py
from dataclasses import dataclass
from typing import List, Dict
import json

@dataclass
class EvalQuestion:
    question: str
    expected_answer: str
    expected_sources: List[str]

class RAGEvaluator:
    def __init__(self, rag_chain):
        self.rag_chain = rag_chain
        
    def evaluate_questions(self, eval_questions: List[EvalQuestion]):
        results = []
        
        for eq in eval_questions:
            # Get answer
            result = self.rag_chain({"query": eq.question})
            
            # Evaluate answer quality
            answer_similarity = self._calculate_similarity(
                result['result'], 
                eq.expected_answer
            )
            
            # Check if correct sources were used
            sources_used = [
                doc.metadata.get('source', '') 
                for doc in result['source_documents']
            ]
            source_accuracy = self._calculate_source_accuracy(
                sources_used, 
                eq.expected_sources
            )
            
            results.append({
                'question': eq.question,
                'answer_similarity': answer_similarity,
                'source_accuracy': source_accuracy,
                'response_time': result.get('response_time', 0)
            })
        
        return results
    
    def _calculate_similarity(self, answer1, answer2):
        # Use embedding similarity or other metrics
        # Simplified version here
        return len(set(answer1.split()) & set(answer2.split())) / len(set(answer1.split()))
    
    def _calculate_source_accuracy(self, found_sources, expected_sources):
        if not expected_sources:
            return 1.0
        
        correct = sum(1 for s in expected_sources if any(s in fs for fs in found_sources))
        return correct / len(expected_sources)