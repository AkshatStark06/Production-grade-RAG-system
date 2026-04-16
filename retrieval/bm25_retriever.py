from rank_bm25 import BM25Okapi
from typing import List, Tuple


class BM25Retriever:
    """
    BM25-based keyword retriever.
    """

    def __init__(self, documents: List[str]):
        """
        Initialize BM25 with tokenized documents.
        """
        self.documents = documents
        self.tokenized_docs = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search documents using BM25.

        Returns:
            List of (document, score)
        """
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        return [(self.documents[i], scores[i]) for i in ranked_indices]