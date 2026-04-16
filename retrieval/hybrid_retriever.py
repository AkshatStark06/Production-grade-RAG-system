from typing import List
from config.config_loader import load_config


class HybridRetriever:
    """
    Combines BM25 and Vector search results.
    """

    def __init__(self, bm25, vector_store, embedding_model):
        self.bm25 = bm25
        self.vector_store = vector_store
        self.embedding_model = embedding_model

        # ✅ Load config ONCE
        config = load_config("config/settings.yaml")

        # ✅ Decide top_k based on mode
        if config["mode"] == "ci":
            self.top_k = config["ci_retrieval"]["top_k"]
        else:
            self.top_k = config["retrieval"]["top_k"]

    def search(self, query: str) -> List[str]:
        """
        Hybrid retrieval.

        Steps:
        1. BM25 search
        2. Vector search
        3. Merge results
        """

        top_k = self.top_k  # ✅ Use controlled value

        # BM25 results
        bm25_results = self.bm25.search(query, top_k)

        # Vector results
        query_embedding = self.embedding_model.encode([query])
        vector_results = self.vector_store.search(query_embedding, top_k)

        # Combine texts
        combined = []

        for text, _ in bm25_results:
            combined.append(text)

        for text, _ in vector_results:
            combined.append(text)

        # Remove duplicates
        unique_results = list(dict.fromkeys(combined))

        return unique_results[:top_k]