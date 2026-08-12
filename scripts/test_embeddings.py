from app.embeddings import EmbeddingModel


model = EmbeddingModel()

texts = [
    "Python classes provide a means of bundling data and functionality together.",
    "A Python function is a block of reusable code."
]

embeddings = model.encode(texts)

print(f"Model: {model.model_name}")
print(f"Embedding dimension: {model.dimension}")
print(f"Number of embeddings: {len(embeddings)}")
print(f"First embedding length: {len(embeddings[0])}")
print(f"First 5 values: {embeddings[0][:5]}")