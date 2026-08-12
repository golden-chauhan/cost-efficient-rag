import json
from pathlib import Path


QUESTIONS_FILE = Path("evaluation/questions.json")
CHUNKS_FILE = Path("data/chunks.json")


questions = json.loads(
    QUESTIONS_FILE.read_text(encoding="utf-8")
)

chunks = json.loads(
    CHUNKS_FILE.read_text(encoding="utf-8")
)


for question in questions:

    print("\n" + "=" * 70)
    print(question["id"])
    print(question["question"])
    print("Expected source:", question["relevant_source"])
    print("=" * 70)

    source_chunks = [
        chunk
        for chunk in chunks
        if chunk["source"] == question["relevant_source"]
    ]

    # Display chunks containing important query terms.
    query_words = [
        word.lower().strip("?,.!:")
        for word in question["question"].split()
        if len(word) > 3
    ]

    scored = []

    for chunk in source_chunks:

        text = chunk["text"].lower()

        score = sum(
            1 for word in query_words
            if word in text
        )

        if score > 0:
            scored.append((score, chunk))

    scored.sort(
        key=lambda x: x[0],
        reverse=True
    )

    for score, chunk in scored[:5]:

        print(
            f"\nScore: {score}"
        )

        print(
            f"Chunk index: {chunk['chunk_index']}"
        )

        print(
            f"ID: {chunk['id']}"
        )

        print(
            f"Text: {chunk['text'][:500]}"
        )