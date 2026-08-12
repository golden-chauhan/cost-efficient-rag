from pathlib import Path

from app.loaders import load_document
from app.chunker import chunk_text
from app.embeddings import EmbeddingModel
from app.vector_store import VectorStore
from app.utils import generate_chunk_id


# --------------------------------------------------
# 1. Load document
# --------------------------------------------------

file_path = Path("documents/classes.html")

text = load_document(file_path)

print(f"Document: {file_path.name}")
print(f"Characters: {len(text)}")


# --------------------------------------------------
# 2. Chunk document
# --------------------------------------------------

chunks = chunk_text(
    text,
    chunk_size=500,
    chunk_overlap=100
)

print(f"Chunks: {len(chunks)}")


# --------------------------------------------------
# 3. Generate embeddings
# --------------------------------------------------

embedding_model = EmbeddingModel()

embeddings = embedding_model.encode(chunks)

print(f"Embedding dimension: {embedding_model.dimension}")


# --------------------------------------------------
# 4. Generate deterministic IDs
# --------------------------------------------------

ids = [
    generate_chunk_id(
        file_path.name,
        index,
        chunk
    )
    for index, chunk in enumerate(chunks)
]


# --------------------------------------------------
# 5. Create metadata
# --------------------------------------------------

metadatas = [
    {
        "source": file_path.name,
        "document_type": "html",
        "chunk_index": index
    }
    for index in range(len(chunks))
]


# --------------------------------------------------
# 6. Store in ChromaDB
# --------------------------------------------------

vector_store = VectorStore()

vector_store.add_chunks(
    ids=ids,
    documents=chunks,
    embeddings=embeddings,
    metadatas=metadatas
)


# --------------------------------------------------
# 7. Display results
# --------------------------------------------------

print(f"Vector store count: {vector_store.count()}")

print("\nFirst stored chunk:")
print(chunks[0])