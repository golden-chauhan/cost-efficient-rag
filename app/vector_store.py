from pathlib import Path

import chromadb


CHROMA_PATH = Path("data/chroma")
COLLECTION_NAME = "python_docs"


class VectorStore:
    def __init__(
        self,
        path: str | Path = CHROMA_PATH,
        collection_name: str = COLLECTION_NAME
    ):
        self.path = str(path)

        self.client = chromadb.PersistentClient(
            path=self.path
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "Python documentation RAG corpus"
            }
        )

    def add_chunks(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict]
    ) -> None:
        """Insert or update chunks in ChromaDB."""

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def count(self) -> int:
        """Return the number of stored chunks."""
        return self.collection.count()

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict | None = None
    ) -> dict:
        """Search the vector database."""

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where
        )