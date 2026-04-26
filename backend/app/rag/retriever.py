from pathlib import Path
from typing import Optional

import faiss
from sentence_transformers import SentenceTransformer


FAISS_INDEX_PATH = Path(__file__).resolve().parents[3] / "models" / "faiss_index"
INDEX_FILE = FAISS_INDEX_PATH / "index.faiss"
CHUNKS_FILE = FAISS_INDEX_PATH / "chunks.txt"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 4

_index: Optional[faiss.IndexFlatL2] = None
_chunks: Optional[list[str]] = None
_embedder: Optional[SentenceTransformer] = None


def _load():
    global _index, _chunks, _embedder

    if _index is None:
        if not INDEX_FILE.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {INDEX_FILE}. "
                "Run backend/app/rag/ingest.py first."
            )
        _index = faiss.read_index(str(INDEX_FILE))

    if _chunks is None:
        _chunks = CHUNKS_FILE.read_text(encoding="utf-8").split("\n<<<CHUNK>>>\n")

    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)


# Maximum L2 distance to still consider a chunk relevant.
# all-MiniLM-L6-v2 produces unit-normalised vectors; cosine distance 1.0
# corresponds to L2 distance ~sqrt(2) ≈ 1.41. Distances above this threshold
# mean the query has essentially no overlap with any indexed content.
OUT_OF_SCOPE_THRESHOLD = 1.2


def retrieve(query: str, top_k: int = TOP_K) -> tuple[list[str], float]:
    """Return (chunks, best_distance). best_distance is the L2 distance of
    the closest chunk; higher means less relevant."""
    _load()

    query_vector = _embedder.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = _index.search(query_vector, top_k)

    best_distance = float(distances[0][0])
    chunks = [_chunks[i] for i in indices[0] if i < len(_chunks)]
    return chunks, best_distance



