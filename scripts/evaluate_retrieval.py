import json
import math
from pathlib import Path

from app.retriever import Retriever


GROUND_TRUTH_FILE = Path(
    "evaluation/ground_truth.json"
)

OUTPUT_FILE = Path(
    "results/retrieval_evaluation.json"
)

TOP_K = 5


def reciprocal_rank(retrieved, relevant):
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / rank

    return 0.0


def recall_at_k(retrieved, relevant, k):
    if not relevant:
        return 0.0

    retrieved_k = set(retrieved[:k])
    relevant_set = set(relevant)

    return len(
        retrieved_k.intersection(relevant_set)
    ) / len(relevant_set)


def ndcg_at_k(retrieved, relevant, k):
    relevant_set = set(relevant)

    dcg = 0.0

    for rank, chunk_id in enumerate(
        retrieved[:k],
        start=1
    ):
        if chunk_id in relevant_set:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_relevant = min(
        len(relevant_set),
        k
    )

    if ideal_relevant == 0:
        return 0.0

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_relevant + 1)
    )

    return dcg / idcg


# --------------------------------------------------
# Load ground truth
# --------------------------------------------------

ground_truth = json.loads(
    GROUND_TRUTH_FILE.read_text(
        encoding="utf-8"
    )
)


retriever = Retriever()

results = []

recall_1 = []
recall_3 = []
recall_5 = []
mrr_scores = []
ndcg_5 = []


# --------------------------------------------------
# Evaluate every question
# --------------------------------------------------

for item in ground_truth:

    question_id = item["id"]
    question = item["question"]
    relevant = item["relevant_chunks"]

    search_result = retriever.retrieve(
        query=question,
        top_k=TOP_K
    )

    retrieved_ids = search_result["ids"][0]

    r1 = recall_at_k(
        retrieved_ids,
        relevant,
        1
    )

    r3 = recall_at_k(
        retrieved_ids,
        relevant,
        3
    )

    r5 = recall_at_k(
        retrieved_ids,
        relevant,
        5
    )

    rr = reciprocal_rank(
        retrieved_ids,
        relevant
    )

    ndcg = ndcg_at_k(
        retrieved_ids,
        relevant,
        5
    )

    recall_1.append(r1)
    recall_3.append(r3)
    recall_5.append(r5)
    mrr_scores.append(rr)
    ndcg_5.append(ndcg)

    results.append(
        {
            "id": question_id,
            "question": question,
            "relevant_chunks": relevant,
            "retrieved_chunks": retrieved_ids,
            "recall_at_1": r1,
            "recall_at_3": r3,
            "recall_at_5": r5,
            "reciprocal_rank": rr,
            "ndcg_at_5": ndcg
        }
    )


# --------------------------------------------------
# Calculate averages
# --------------------------------------------------

count = len(results)

summary = {
    "questions": count,
    "recall_at_1": sum(recall_1) / count,
    "recall_at_3": sum(recall_3) / count,
    "recall_at_5": sum(recall_5) / count,
    "mrr": sum(mrr_scores) / count,
    "ndcg_at_5": sum(ndcg_5) / count
}


# --------------------------------------------------
# Save results
# --------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

output = {
    "configuration": {
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_dimension": 384,
        "chunk_size": 500,
        "chunk_overlap": 100,
        "top_k": 5,
        "vector_store": "ChromaDB"
    },
    "summary": summary,
    "questions": results
}

OUTPUT_FILE.write_text(
    json.dumps(
        output,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# --------------------------------------------------
# Print results
# --------------------------------------------------

print("\n" + "=" * 60)
print("RETRIEVAL EVALUATION")
print("=" * 60)

print(f"Questions:   {count}")
print(f"Recall@1:    {summary['recall_at_1']:.4f}")
print(f"Recall@3:    {summary['recall_at_3']:.4f}")
print(f"Recall@5:    {summary['recall_at_5']:.4f}")
print(f"MRR:         {summary['mrr']:.4f}")
print(f"nDCG@5:      {summary['ndcg_at_5']:.4f}")

print("\nSaved to:")
print(OUTPUT_FILE)