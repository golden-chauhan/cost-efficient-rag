from pathlib import Path

from app.chunker import chunk_text
from app.embeddings import EmbeddingModel
from app.loaders import load_document
from app.utils import generate_chunk_id
from app.vector_store import VectorStore


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DOCUMENTS_DIR = Path("documents")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


# --------------------------------------------------
# Main ingestion pipeline
# --------------------------------------------------

def ingest_documents():

    embedding_model = EmbeddingModel()
    vector_store = VectorStore()

    total_documents = 0
    total_chunks = 0

    print("=" * 70)
    print("RAG DOCUMENT INGESTION")
    print("=" * 70)

    for file_path in sorted(DOCUMENTS_DIR.iterdir()):

        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in {
            ".pdf",
            ".html",
            ".htm",
            ".md",
            ".markdown"
        }:
            continue

        print(f"\nProcessing: {file_path.name}")

        # ------------------------------------------
        # 1. Load document
        # ------------------------------------------

        text = load_document(file_path)

        print(f"Characters: {len(text):,}")

        # ------------------------------------------
        # 2. Chunk document
        # ------------------------------------------

        chunks = chunk_text(
            text,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        print(f"Chunks: {len(chunks)}")

        if not chunks:
            print("Skipping empty document.")
            continue

        # ------------------------------------------
        # 3. Generate embeddings
        # ------------------------------------------

        embeddings = embedding_model.encode(chunks)

        # ------------------------------------------
        # 4. Generate deterministic IDs
        # ------------------------------------------

        ids = [
            generate_chunk_id(
                file_path.name,
                index,
                chunk
            )
            for index, chunk in enumerate(chunks)
        ]

        # ------------------------------------------
        # 5. Create metadata
        # ------------------------------------------

        document_type = file_path.suffix.lower().replace(".", "")

        metadatas = [
            {
                "source": file_path.name,
                "document_type": document_type,
                "chunk_index": index,
            }
            for index in range(len(chunks))
        ]

        # ------------------------------------------
        # 6. Store in ChromaDB
        # ------------------------------------------

        vector_store.add_chunks(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )

        total_documents += 1
        total_chunks += len(chunks)

        print("Stored successfully.")

    # --------------------------------------------------
    # Final statistics
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("INGESTION COMPLETE")
    print("=" * 70)

    print(f"Documents processed: {total_documents}")
    print(f"Chunks processed:    {total_chunks}")
    print(f"Embedding dimension: {embedding_model.dimension}")
    print(f"Vector store count:  {vector_store.count()}")
    print("=" * 70)


if __name__ == "__main__":
    ingest_documents()