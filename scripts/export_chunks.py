from pathlib import Path
import json

from app.chunker import chunk_text
from app.loaders import load_document
from app.utils import generate_chunk_id


DOCUMENTS_DIR = Path("documents")


all_chunks = []


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

    text = load_document(file_path)

    chunks = chunk_text(
        text,
        chunk_size=500,
        chunk_overlap=100
    )

    for index, chunk in enumerate(chunks):

        chunk_id = generate_chunk_id(
            file_path.name,
            index,
            chunk
        )

        all_chunks.append({
            "id": chunk_id,
            "source": file_path.name,
            "chunk_index": index,
            "text": chunk
        })


output_path = Path("data/chunks.json")

output_path.write_text(
    json.dumps(
        all_chunks,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


print(f"Exported {len(all_chunks)} chunks.")
print(f"Saved to: {output_path}")