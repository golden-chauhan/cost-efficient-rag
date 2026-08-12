from app.rag import RAGPipeline


def main():

    rag = RAGPipeline()

    questions = [
        "What is the capital of France?"
    ]

    for question in questions:

        print("\n" + "=" * 70)
        print("QUESTION")
        print("=" * 70)
        print(question)

        result = rag.query(
            question=question,
            top_k=5
        )

        print("\nANSWER:")
        print(result["answer"])

        print("\nCITATIONS:")

        for citation in result["citations"]:
            print(
                f"[{citation['citation']}] "
                f"{citation['source']} "
                f"(chunk {citation['chunk_index']}, "
                f"distance={citation['distance']})"
            )

        print("\nMETRICS:")

        print(
            "Retrieved chunks:",
            result["retrieved_chunks"]
        )

        print(
            "Retrieval latency:",
            result["retrieval_latency_ms"],
            "ms"
        )

        print(
            "Generation latency:",
            result.get(
                "generation_latency_ms",
                0
            ),
            "ms"
        )

        print(
            "Total latency:",
            result["total_latency_ms"],
            "ms"
        )

        print(
            "Input tokens:",
            result["input_tokens"]
        )

        print(
            "Output tokens:",
            result["output_tokens"]
        )

        print(
            "Total tokens:",
            result["total_tokens"]
        )


if __name__ == "__main__":
    main()