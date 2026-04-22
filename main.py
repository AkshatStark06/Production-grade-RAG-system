from pipeline import RAGPipeline


# ✅ ADD THIS FUNCTION (VERY IMPORTANT)
def build_rag_pipeline():
    return RAGPipeline()


def main():
    rag = RAGPipeline()

    query = input("\n❓ Enter your question: ")

    result = rag.run(query)

    print("\n💡 FINAL ANSWER:\n")
    print(result["answer"])


if __name__ == "__main__":
    main()