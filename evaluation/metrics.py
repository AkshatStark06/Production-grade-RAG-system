from sentence_transformers import SentenceTransformer, util


class Metrics:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def cosine_similarity(self, text1, text2):
        emb1 = self.model.encode(text1, convert_to_tensor=True)
        emb2 = self.model.encode(text2, convert_to_tensor=True)

        return util.cos_sim(emb1, emb2).item()

    # ----------------------------
    # 1. Answer Correctness
    # ----------------------------
    def answer_correctness(self, generated_answer, ground_truth):
        return self.cosine_similarity(generated_answer, ground_truth)

    # ----------------------------
    # 2. Faithfulness
    # ----------------------------
    def faithfulness(self, generated_answer, context):
        return self.cosine_similarity(generated_answer, context)

    # ----------------------------
    # 3. Retrieval Hit
    # ----------------------------
    def retrieval_hit(self, ground_truth, retrieved_chunks, threshold=0.4):
        for chunk in retrieved_chunks:
            score = self.cosine_similarity(ground_truth, chunk)
            if score >= threshold:
                return 1
        return 0