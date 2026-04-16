from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np


class EmbeddingModel:
    """
    Wrapper around sentence-transformers model
    for generating embeddings.
    """

    def __init__(self, model_name: str):
        """
        Initialize embedding model.

        Args:
            model_name (str): HuggingFace model name
        """
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Convert list of texts into embeddings.

        Args:
            texts (List[str]): Input text chunks

        Returns:
            np.ndarray: Embedding vectors
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        return embeddings