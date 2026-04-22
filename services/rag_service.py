from main import build_rag_pipeline
from config.config_loader import load_config
from utils.logger import logger
from functools import lru_cache
import hashlib
import time

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
    
    def stream_query(self, user_query: str):
        result = self.pipeline.run(user_query)
        answer = result["answer"]

        words = answer.split()

        for word in words:
            time.sleep(0.08)  # 👈 this creates typing effect
            yield word + " "

    def query(self, user_query: str, use_cache: bool = True):

        logger.info(f"Query received: {user_query}")
        
        query_hash = _hash_query(user_query)    
        
        # Store query for cache lookup
        self._query_store[query_hash] = user_query
        
        # Choose cached vs fresh
        if use_cache:
            result = self._cached_pipeline_run(query_hash)
        else:
            result = self.pipeline.run(user_query)

        # ✅ FIRST extract
        answer = result["answer"]
        contexts = result.get("contexts", [])

        # ✅ THEN debug print  
        for i, ctx in enumerate(contexts):
            print(f"\n[CTX {i+1}]: {ctx[:200]}")        

        logger.info("Answer generated successfully")

        return {
            "query": user_query,
            "answer": answer,
            "context": contexts
        }
