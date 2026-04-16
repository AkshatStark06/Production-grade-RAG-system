from sentence_transformers import CrossEncoder
from typing import List, Tuple


class CrossEncoderReranker:
    def __init__(self, model_name: str, top_k: int = 5):
        """
        Cross-Encoder Re-Ranker

        Args:
            model_name (str): HuggingFace model name
            top_k (int): number of final results to return
        """
        self.model = CrossEncoder(model_name)
        self.top_k = top_k

    def rerank(self, query: str, documents: List[str]) -> List[Tuple[str, float]]:
        """
        Re-rank documents based on relevance to query

        Args:
            query (str): user query
            documents (List[str]): retrieved chunks

        Returns:
            List of (document, score) sorted by relevance
        """

        # Create (query, doc) pairs
        pairs = [(query, doc) for doc in documents]

        # Get relevance scores
        scores = self.model.predict(pairs)

        # Combine docs with scores
        doc_score_pairs = list(zip(documents, scores))

        # Sort by score (descending)
        ranked = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

        # Return top_k
        return ranked[:self.top_k]