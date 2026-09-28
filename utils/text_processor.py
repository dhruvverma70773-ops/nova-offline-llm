import re


def clean_text(text: str) -> str:
    """Normalize whitespace/control characters only.

    NOTE: earlier version tokenized with NLTK and tried to strip stopwords,
    but the filter condition (`not a stopword OR alphanumeric`) is true for
    almost every real word, so it silently did nothing useful while still
    requiring an internet download of NLTK corpora on first run -- which
    defeats the purpose of an *offline* assistant. Worse, if it had worked,
    stripping stopwords before indexing/answering actively hurts retrieval
    and makes LLM answers read like garbled keyword soup. Plain whitespace
    normalization is what we actually want here.
    """
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_text(text: str, chunk_size: int = 700, overlap: int = 100):
    """Split text into overlapping word-chunks for embedding."""
    if not text:
        return []
    words = text.split()
    if not words:
        return []
    overlap = max(0, min(overlap, chunk_size - 1))
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap
    return chunks
