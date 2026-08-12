import time

from app.retriever import Retriever
from app.generator import LLMGenerator


class RAGPipeline:

    def __init__(self):
        self.retriever = Retriever()
        self.generator = LLMGenerator()

    def query(
        self,
        question: str,
        top_k: int = 5,
        where: dict | None = None
    ) -> dict:

        # --------------------------------------------------
        # 1. RETRIEVAL
        # --------------------------------------------------

        retrieval_start = time.perf_counter()

        retrieved = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            where=where
        )

        retrieval_latency_ms = (
            time.perf_counter() - retrieval_start
        ) * 1000

        # --------------------------------------------------
        # 2. PREPARE CONTEXT
        # --------------------------------------------------

        contexts = []

        for result in retrieved:

            contexts.append({
                "text": result["text"],
                "source": result["source"],
                "chunk_index": result["chunk_index"],
                "distance": result["distance"]
            })

        # --------------------------------------------------
        # 3. GENERATE ANSWER
        # --------------------------------------------------

        generation_start = time.perf_counter()

        generation_result = self.generator.generate(
            question=question,
            contexts=contexts
        )

        generation_latency_ms = (
            time.perf_counter() - generation_start
        ) * 1000

        # --------------------------------------------------
        # 4. TOTAL LATENCY
        # --------------------------------------------------

        total_latency_ms = (
            retrieval_latency_ms
            + generation_latency_ms
        )

        # --------------------------------------------------
        # 5. CITATIONS
        # --------------------------------------------------

        citations = []

        for i, context in enumerate(contexts, start=1):

            citations.append({
                "citation": i,
                "source": context["source"],
                "chunk_index": context["chunk_index"],
                "distance": context["distance"]
            })

        # --------------------------------------------------
        # 6. FINAL RESULT
        # --------------------------------------------------

        return {
            "question": question,

            "answer": generation_result["answer"],

            "citations": citations,

            "retrieved_chunks": len(contexts),

            "retrieval_latency_ms": round(
                retrieval_latency_ms,
                2
            ),

            "generation_latency_ms": round(
                generation_latency_ms,
                2
            ),

            "total_latency_ms": round(
                total_latency_ms,
                2
            ),

            "input_tokens": generation_result[
                "input_tokens"
            ],

            "output_tokens": generation_result[
                "output_tokens"
            ],

            "total_tokens": generation_result[
                "total_tokens"
            ]
        }