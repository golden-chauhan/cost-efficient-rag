from pathlib import Path
from app.loaders import load_document


documents_dir = Path("documents")

for file_path in documents_dir.iterdir():
    if file_path.is_file():
        text = load_document(file_path)

        print("=" * 60)
        print(f"FILE: {file_path.name}")
        print(f"CHARACTERS: {len(text)}")
        print(f"PREVIEW:\n{text[:500]}")