from main import build_rag_pipeline
from config.config_loader import load_config
from utils.logger import logger
from functools import lru_cache
import hashlib

def _hash_query(user_query: str) -> str:
    return hashlib.md5(user_query.strip().lower().encode()).hexdigest()

class RAGService:
    def __init__(self):
        logger.info("Initializing RAG Service...")
        self._query_store = {}
        self.config = load_config()
        self.pipeline = build_rag_pipeline()

    @lru_cache(maxsize=100)
    def _cached_pipeline_run(self, query_hash: str):
        return self.pipeline.run(self._query_store[query_hash])

    def query(self, user_query: str):
        logger.info(f"Query received: {user_query}")
        query_hash = _hash_query(user_query)
        self._query_store[query_hash] = user_query
        result = self._cached_pipeline_run(query_hash)

        answer = result["answer"]
        contexts = result.get("contexts", [])

        logger.info("Answer generated successfully")

        return {
            "query": user_query,
            "answer": answer,
            "context": contexts
        }
