import hashlib


def generate_chunk_id(
    document_name: str,
    chunk_index: int,
    chunk_text: str
) -> str:
    """
    Generate a deterministic ID for a document chunk.
    """

    content = (
        f"{document_name}|"
        f"{chunk_index}|"
        f"{chunk_text}"
    )

    digest = hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()[:16]

    return f"{document_name}_{chunk_index}_{digest}"