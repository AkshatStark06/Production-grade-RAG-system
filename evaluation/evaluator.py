import json
from evaluation.metrics import Metrics


class Evaluator:
    def __init__(self, rag_pipeline, dataset_path):
        self.rag = rag_pipeline
        self.dataset_path = dataset_path
        self.metrics = Metrics()

    def load_dataset(self):
        with open(self.dataset_path, "r") as f:
            return json.load(f)

    def evaluate(self):
        self.data = self.load_dataset()

        total = len(self.data)

        correctness_scores = []
        faithfulness_scores = []
        retrieval_hits = 0

        for item in self.data:
            query = item["question"]
            ground_truth = item["answer"]

            # 🔥 Run your pipeline
            result = self.rag.run(query)

            generated_answer = result["answer"]
            context_chunks = result.get("context") or result.get("contexts", [])  # list of strings

            # ---------------- Metrics ----------------
            correctness = self.metrics.answer_correctness(
                generated_answer, ground_truth
            )

            faithfulness = self.metrics.faithfulness(
                generated_answer, " ".join(context_chunks)
            )

            hit = self.metrics.retrieval_hit(
                ground_truth, context_chunks
            )

            correctness_scores.append(correctness)
            faithfulness_scores.append(faithfulness)
            retrieval_hits += hit

            print("\n==============================")
            print(f"Query: {query}")
            print(f"Ground Truth: {ground_truth}")

            print("\nTop Retrieved Chunks:")
            for c in context_chunks:
                print("-", c[:150])

            print("\nGenerated Answer:")
            print(generated_answer)
            print("==============================\n")

        # ---------------- Final Scores ----------------
        results = {
            "total_samples": len(self.data),
            "correctness": sum(correctness_scores) / total,
            "faithfulness": sum(faithfulness_scores) / total,
            "hit_rate": retrieval_hits / total,
        }

        return results


# ✅ THIS PART IS REQUIRED FOR CI/CD
if __name__ == "__main__":
    # ⚠️ Make sure this function exists in your main.py
    from main import build_rag_pipeline

    # Build RAG pipeline
    rag_pipeline = build_rag_pipeline()

    # Initialize evaluator
    evaluator = Evaluator(
        rag_pipeline=rag_pipeline,
        dataset_path="data/eval_dataset.json"
    )

    # Run evaluation
    results = evaluator.evaluate()

    # Print results (for logs)
    print("\n📊 FINAL RESULTS:")
    print(json.dumps(results, indent=2))

    # Save results (for CI threshold check)
    with open("evaluation/results.json", "w") as f:
        json.dump(results, f)