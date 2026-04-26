"""
Ingest script: chunks the medical knowledge base, embeds the chunks,
and saves a FAISS index to models/faiss_index/.

Run from the project root:
    python backend/app/rag/ingest.py
"""

from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


SOURCES_DIR = Path("data/rag_sources")
OUTPUT_DIR = Path("models/faiss_index")
INDEX_FILE = OUTPUT_DIR / "index.faiss"
CHUNKS_FILE = OUTPUT_DIR / "chunks.txt"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    # Split on the explicit section separator first, then word-window each section
    sections = [s.strip() for s in text.split("---") if s.strip()]
    chunks = []
    for section in sections:
        words = section.split()
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk.strip())
            if end == len(words):
                break
            start += chunk_size - overlap
    return chunks


def main():
    source_files = sorted(SOURCES_DIR.glob("*.txt"))
    if not source_files:
        raise FileNotFoundError(f"No .txt files found in: {SOURCES_DIR}")

    chunks: list[str] = []
    for src in source_files:
        print(f"Reading: {src}")
        text = src.read_text(encoding="utf-8")
        chunks.extend(chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP))

    print(f"Total chunks: {len(chunks)}")

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} ...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Embedding chunks ...")
    embeddings = embedder.encode(chunks, convert_to_numpy=True).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_FILE))
    CHUNKS_FILE.write_text("\n<<<CHUNK>>>\n".join(chunks), encoding="utf-8")

    print(f"Saved FAISS index to: {INDEX_FILE}")
    print(f"Saved {len(chunks)} chunks to: {CHUNKS_FILE}")


if __name__ == "__main__":
    main()
