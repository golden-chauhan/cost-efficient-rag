from app.retriever import Retriever


retriever = Retriever()


queries = [
    "What is a Python class?",
    "How does inheritance work in Python?",
    "What is a lambda expression?",
    "How does Python handle exceptions?",
    "What is the purpose of the try statement?",
    "How can a list be used as a stack?",
    "What is the difference between a list and a tuple?",
    "How are dictionaries used in Python?",
    "How do you read and write files in Python?",
    "How are modules imported in Python?",
]


for query in queries:

    print("\n" + "=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)

    results = retriever.retrieve(
        query=query,
        top_k=5
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1
    ):

        print(f"\n--- RESULT {i} ---")

        print(f"Source: {metadata['source']}")
        print(f"Chunk: {metadata['chunk_index']}")
        print(f"ID: {results['ids'][0][i-1]}")
        print(f"Distance: {distance:.4f}")

        print("\nText:")
        print(document[:700])