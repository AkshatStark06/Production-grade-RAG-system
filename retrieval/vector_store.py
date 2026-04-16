import faiss
import numpy as np
import os
import pickle
from typing import List, Tuple


class FAISSVectorStore:
    """
    FAISS-based vector store for similarity search.
    """

    def __init__(self, dimension: int):
        """
        Initialize FAISS index.

        Args:
            dimension (int): Embedding dimension
        """
        self.index = faiss.IndexFlatL2(dimension)
        self.text_chunks = []

    def add_embeddings(self, embeddings: np.ndarray, texts: List[str]):
        """
        Add embeddings and corresponding texts.

        Args:
            embeddings (np.ndarray): Embedding vectors
            texts (List[str]): Original text chunks
        """
        self.index.add(embeddings)
        self.text_chunks.extend(texts)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for most similar chunks.

        Args:
            query_embedding (np.ndarray): Query vector
            top_k (int): Number of results

        Returns:
            List of (text, distance)
        """
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.text_chunks):
                results.append((self.text_chunks[idx], distances[0][i]))

        return results

    def save(self, path: str):
        """
        Save FAISS index and metadata.
        """
        os.makedirs(path, exist_ok=True)

        faiss.write_index(self.index, os.path.join(path, "index.faiss"))

        with open(os.path.join(path, "texts.pkl"), "wb") as f:
            pickle.dump(self.text_chunks, f)

    def load(self, path: str):
        """
        Load FAISS index and metadata.
        """
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))

        with open(os.path.join(path, "texts.pkl"), "rb") as f:
            self.text_chunks = pickle.load(f)