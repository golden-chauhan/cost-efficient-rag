import re


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> list[str]:
    """
    Create sentence-aware overlapping chunks.

    Chunks are built from complete sentences whenever possible.
    Overlap is also sentence-based rather than character-based.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    text = text.strip()

    if not text:
        return []

    sentences = split_into_sentences(text)

    chunks = []
    current_sentences = []
    current_length = 0

    for sentence in sentences:

        sentence_length = len(sentence)

        # If adding this sentence keeps the chunk within the limit
        proposed_length = (
            current_length
            + sentence_length
            + (1 if current_sentences else 0)
        )

        if proposed_length <= chunk_size:
            current_sentences.append(sentence)
            current_length = proposed_length

        else:
            # Save current chunk
            if current_sentences:
                chunks.append(" ".join(current_sentences))

            # Start next chunk with the current sentence
            current_sentences = [sentence]
            current_length = sentence_length

    # Save final chunk
    if current_sentences:
        chunks.append(" ".join(current_sentences))

    # --------------------------------------------------
    # Sentence-aware overlap
    # --------------------------------------------------

    if chunk_overlap > 0 and len(chunks) > 1:

        overlapped_chunks = [chunks[0]]

        for i in range(1, len(chunks)):

            previous_sentences = split_into_sentences(
                chunks[i - 1]
            )

            overlap_sentences = []
            overlap_length = 0

            # Take complete sentences from the end of the
            # previous chunk until the overlap limit is reached.
            for sentence in reversed(previous_sentences):

                sentence_length = len(sentence)

                if (
                    overlap_length + sentence_length
                    + (1 if overlap_sentences else 0)
                    <= chunk_overlap
                ):
                    overlap_sentences.insert(0, sentence)

                    overlap_length += (
                        sentence_length
                        + (1 if overlap_sentences else 0)
                    )
                else:
                    break

            if overlap_sentences:
                combined = (
                    " ".join(overlap_sentences)
                    + " "
                    + chunks[i]
                )
            else:
                combined = chunks[i]

            overlapped_chunks.append(combined)

        chunks = overlapped_chunks

    return chunks