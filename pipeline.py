from ingestion.document_loader import load_documents
from ingestion.text_splitter import split_text

from retrieval.embedding_model import EmbeddingModel
from retrieval.vector_store import FAISSVectorStore
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.query_processor import split_query

from reranker.cross_encoder_reranker import CrossEncoderReranker
from llm.llm_generator import LLMGenerator

from config.config_loader import load_config


class RAGPipeline:
    def __init__(self):
        # Load config
        self.config = load_config("config/settings.yaml")

        # Load + chunk docs
        documents = load_documents("data/sample.txt")

        self.chunks = split_text(
            documents,
            chunk_size=self.config["chunking"]["chunk_size"],
            overlap=self.config["chunking"]["overlap"]
        )

        # Embeddings
        self.embedding_model = EmbeddingModel(
            self.config["embedding"]["model_name"]
        )
        embeddings = self.embedding_model.encode(self.chunks)

        # Vector DB
        dimension = embeddings.shape[1]
        self.vector_store = FAISSVectorStore(dimension)
        self.vector_store.add_embeddings(embeddings, self.chunks)

        # BM25
        self.bm25 = BM25Retriever(self.chunks)

        # Hybrid
        self.hybrid = HybridRetriever(
            self.bm25,
            self.vector_store,
            self.embedding_model
        )

        # Reranker
        self.reranker = CrossEncoderReranker(
            model_name=self.config["reranker"]["model_name"],
            top_k=self.config["reranker"]["top_k"]
        )

        # LLM
        self.llm = LLMGenerator()

    # 🔥 THIS IS WHAT EVALUATION NEEDS
    def run(self, query):

        sub_queries = split_query(query)

        final_answers = []
        all_contexts = []
        confidence_scores = []

        for q in sub_queries:

            retrieved_docs = self.hybrid.search(q)
            reranked = self.reranker.rerank(q, retrieved_docs)

            if reranked:
                confidence_scores.append(reranked[0][1])

            top_chunks = [doc for doc, _ in reranked[:4]]

            answer = self.llm.generate(top_chunks, q)

            final_answers.append(answer)
            all_contexts.extend(top_chunks)

        avg_confidence = (
            sum(confidence_scores) / len(confidence_scores)
            if confidence_scores else 0.0
        )

        return {
            "answer": " ".join(final_answers),
            "contexts": all_contexts,
            "confidence": float(avg_confidence)
        }