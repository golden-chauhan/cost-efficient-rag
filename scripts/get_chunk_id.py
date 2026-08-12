import json
from pathlib import Path

chunks = json.loads(
    Path("data/chunks.json").read_text(encoding="utf-8")
)

for chunk in chunks:
    if (
        chunk["source"] == "modules.html"
        and chunk["chunk_index"] == 2
    ):
        print("Q17 GOLD CHUNK")
        print("Source:", chunk["source"])
        print("Chunk:", chunk["chunk_index"])
        print("ID:", chunk["id"])
        print("\nText:")
        print(chunk["text"])
        break
else:
    print("Chunk not found.")