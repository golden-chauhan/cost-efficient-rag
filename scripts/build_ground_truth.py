import json
from pathlib import Path

from app.retriever import Retriever


QUESTIONS_FILE = Path("evaluation/questions.json")
OUTPUT_FILE = Path("evaluation/ground_truth.json")


# --------------------------------------------------
# Load questions
# --------------------------------------------------

questions = json.loads(
    QUESTIONS_FILE.read_text(encoding="utf-8")
)

retriever = Retriever()


# --------------------------------------------------
# Existing ground truth we already verified
# --------------------------------------------------

KNOWN_RELEVANT = {
    "Q01": [0],
    "Q02": [53, 54],
    "Q03": [68],
    "Q04": [3, 9],
    "Q05": [8],
    "Q06": [8],
    "Q07": [27],
    "Q08": [32],
    "Q09": [34],
    "Q10": [7],
}


ground_truth = []


# --------------------------------------------------
# Process each question
# --------------------------------------------------

for question in questions:

    qid = question["id"]
    query = question["question"]

    print("\n" + "=" * 80)
    print(f"{qid}: {query}")
    print("=" * 80)

    results = retriever.retrieve(
        query=query,
        top_k=5
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]

    # --------------------------------------------------
    # Display retrieved results
    # --------------------------------------------------

    for i in range(len(documents)):

        print("\n" + "-" * 70)
        print(f"RESULT {i + 1}")
        print("-" * 70)

        print(f"ID:       {ids[i]}")
        print(f"Source:   {metadatas[i]['source']}")
        print(f"Chunk:    {metadatas[i]['chunk_index']}")
        print(f"Distance: {distances[i]:.4f}")

        print("\nText:")
        print(documents[i][:600])

    # --------------------------------------------------
    # Automatically use known ground truth for Q01-Q10
    # --------------------------------------------------

    if qid in KNOWN_RELEVANT:

        relevant_chunk_indexes = KNOWN_RELEVANT[qid]

        relevant_ids = []

        for i, metadata in enumerate(metadatas):

            if metadata["chunk_index"] in relevant_chunk_indexes:
                relevant_ids.append(ids[i])

        print("\nKnown relevant chunk indexes:")
        print(relevant_chunk_indexes)

        print("Retrieved relevant IDs:")
        print(relevant_ids)

    else:

        # --------------------------------------------------
        # Manual selection for remaining questions
        # --------------------------------------------------

        print("\nWhich retrieved results are relevant?")

        print(
            "Enter result numbers separated by commas."
        )

        print(
            "Example: 1,3"
        )

        print(
            "Enter 0 if none of the top-5 results are relevant."
        )

        while True:

            answer = input("> ").strip()

            if answer == "0":
                relevant_ids = []
                break

            try:

                selected = [
                    int(x.strip())
                    for x in answer.split(",")
                ]

                if not selected:
                    raise ValueError

                if any(
                    number < 1 or number > 5
                    for number in selected
                ):
                    raise ValueError

                relevant_ids = [
                    ids[number - 1]
                    for number in selected
                ]

                break

            except ValueError:

                print(
                    "Invalid input. Use something like 1,3 "
                    "or enter 0."
                )

    # --------------------------------------------------
    # Save ground truth
    # --------------------------------------------------

    ground_truth.append(
        {
            "id": qid,
            "question": query,
            "relevant_chunks": relevant_ids
        }
    )


# --------------------------------------------------
# Save file
# --------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE.write_text(
    json.dumps(
        ground_truth,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


print("\n" + "=" * 80)
print("GROUND TRUTH COMPLETE")
print("=" * 80)

print(
    f"Questions processed: {len(ground_truth)}"
)

print(
    f"Saved to: {OUTPUT_FILE}"
)