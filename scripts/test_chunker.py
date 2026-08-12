from pathlib import Path

from app.loaders import load_document
from app.chunker import chunk_text


file_path = Path("documents/classes.html")

text = load_document(file_path)

chunks = chunk_text(
    text,
    chunk_size=500,
    chunk_overlap=100
)

print(f"Original characters: {len(text)}")
print(f"Number of chunks: {len(chunks)}")

print("\n" + "=" * 60)
print("FIRST CHUNK")
print("=" * 60)
print(chunks[0])

print("\n" + "=" * 60)
print("SECOND CHUNK")
print("=" * 60)
print(chunks[1])