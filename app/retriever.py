from app.embeddings import EmbeddingModel
from app.vector_store import VectorStore


class Retriever:

    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        where: dict | None = None
    ) -> list[dict]:
        """
        Retrieve the most relevant chunks for a query.
        """

        # Convert query into an embedding
        query_embedding = self.embedding_model.encode(query)

        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            where=where
        )

        retrieved = []

        # Chroma returns lists inside the result dictionary
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        for i in range(len(documents)):

            metadata = (
                metadatas[i]
                if metadatas and i < len(metadatas)
                else {}
            )

            retrieved.append({
                "id": (
                    ids[i]
                    if ids and i < len(ids)
                    else None
                ),

                "text": documents[i],

                "distance": (
                    distances[i]
                    if distances and i < len(distances)
                    else None
                ),

                "source": metadata.get("source"),

                "chunk_index": metadata.get(
                    "chunk_index"
                )
            })

        return retrieved