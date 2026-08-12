from pathlib import Path
from bs4 import BeautifulSoup
import pymupdf


def load_pdf(file_path: Path) -> str:
    """Extract text from a PDF file."""
    text_parts = []

    with pymupdf.open(file_path) as doc:
        for page in doc:
            text_parts.append(page.get_text())

    return "\n".join(text_parts)


def load_html(file_path: Path) -> str:
    """Extract the main documentation content from an HTML file."""

    html = file_path.read_text(encoding="utf-8")

    soup = BeautifulSoup(html, "html.parser")

    selectors_to_remove = [
        "script",
        "style",
        "nav",
        "header",
        "footer",
        "aside",
        "[role='navigation']",
        ".related",
        ".sphinxsidebar",
        ".sidebar",
        ".theme-switcher",
        ".theme-toggle",
        ".mobile-nav",
        ".mobile-nav-toggle",
    ]

    for selector in selectors_to_remove:
        for element in soup.select(selector):
            element.decompose()

    main_content = (
        soup.find("main")
        or soup.find(attrs={"role": "main"})
        or soup.find("div", class_="document")
    )

    if main_content:
        text = main_content.get_text(separator=" ", strip=True)
    else:
        text = soup.get_text(separator=" ", strip=True)

    # Normalize excessive whitespace.
    text = " ".join(text.split())

    return text

def load_markdown(file_path: Path) -> str:
    """Load Markdown as plain text."""
    return file_path.read_text(encoding="utf-8")


def load_document(file_path: str | Path) -> str:
    """Load a PDF, HTML, or Markdown document based on its extension."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    extension = path.suffix.lower()

    if extension == ".pdf":
        return load_pdf(path)

    if extension in {".html", ".htm"}:
        return load_html(path)

    if extension in {".md", ".markdown"}:
        return load_markdown(path)

    raise ValueError(
        f"Unsupported file type: {extension}. "
        "Supported types: PDF, HTML, Markdown."
    )