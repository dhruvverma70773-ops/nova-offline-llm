from pathlib import Path
from pypdf import PdfReader
from docx import Document

def extract_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
    if path.suffix.lower() == ".docx":
        return "\n".join(p.text for p in Document(str(path)).paragraphs)
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    raise ValueError("Unsupported file type")
