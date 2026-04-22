from evaluation.evaluator import Evaluator
from main import RAGPipeline  # adjust if needed


def main():
    # Initialize your pipeline
    rag = RAGPipeline()

    evaluator = Evaluator(
        rag_pipeline=rag,
        dataset_path="data/eval_dataset.json"
    )

    results = evaluator.evaluate()

    print("\n📊 EVALUATION RESULTS")
    print("=" * 40)
    print(f"Total Samples: {results['total_samples']}")
    print(f"Avg Correctness: {results['correctness']:.4f}")
    print(f"Avg Faithfulness: {results['faithfulness']:.4f}")
    print(f"Retrieval Hit Rate: {results['hit_rate']:.4f}")


if __name__ == "__main__":
    main()