"""Vector embeddings for framework control descriptions.

Used for:
  - Similarity-based cross-framework mapping (e.g., finding the closest 800-53
    controls for a CSF subcategory that lacks an explicit OLIR reference).
  - Semantic search within the catalog at inference time.
  - Candidate generation for the classification model.

Prefer ``sentence-transformers`` (GPU-friendly) when available, falling back to
a lightweight TF-IDF + cosine approach using scikit-learn.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np

from frameworks.catalog import Catalog, ControlEntry

ROOT = Path(__file__).resolve().parent
DEFAULT_EMBEDDINGS_PATH = ROOT / "embeddings.json"
DEFAULT_VECTORIZER_PATH = ROOT / "vectorizer.pkl"


class EmbeddingStore:
    """Manages vector embeddings for every ControlEntry in the catalog."""

    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self._entries = catalog.all_entries()
        self._index: dict[int, ControlEntry] = {i: e for i, e in enumerate(self._entries)}
        self._matrix: np.ndarray | None = None
        self._vectorizer: object = None
        self._vocabulary: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Build / load / save
    # ------------------------------------------------------------------

    def build(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        texts = [e.text_for_embedding() for e in self._entries]
        try:
            self._matrix = _build_with_sentence_transformers(texts, model_name)
            self._vocabulary = {}
        except (ImportError, ValueError, RuntimeError):
            self._matrix, self._vectorizer, self._vocabulary = _build_with_tfidf(texts)

    def save(
        self,
        embeddings_path: Path | str | None = None,
        vectorizer_path: Path | str | None = None,
    ) -> Path:
        target = Path(embeddings_path) if embeddings_path else DEFAULT_EMBEDDINGS_PATH
        if self._matrix is None:
            raise RuntimeError("Embeddings not built. Call `.build()` first.")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "matrix": self._matrix.tolist(),
                    "entry_ids": [f"{e.framework.value}:{e.id}" for e in self._entries],
                    "shape": list(self._matrix.shape),
                },
                ensure_ascii=False,
            )
        )
        if self._vectorizer is not None:
            vp = Path(vectorizer_path) if vectorizer_path else DEFAULT_VECTORIZER_PATH
            vp.write_bytes(pickle.dumps(self._vectorizer))
        return target

    @classmethod
    def load(
        cls,
        catalog: Catalog,
        embeddings_path: Path | str | None = None,
        vectorizer_path: Path | str | None = None,
    ) -> EmbeddingStore:
        embeddings_target = Path(embeddings_path) if embeddings_path else DEFAULT_EMBEDDINGS_PATH
        store = cls(catalog)
        if embeddings_target.exists():
            data = json.loads(embeddings_target.read_text())
            store._matrix = np.array(data["matrix"], dtype=np.float32)

        vp = Path(vectorizer_path) if vectorizer_path else DEFAULT_VECTORIZER_PATH
        if vp.exists():
            store._vectorizer = pickle.loads(vp.read_bytes())

        return store

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
        framework_filter: str | None = None,
    ) -> list[tuple[ControlEntry, float]]:
        if self._matrix is None:
            raise RuntimeError("Embeddings not built or loaded.")
        if self._vectorizer is not None:
            query_vec = self._vectorizer.transform([query]).toarray().astype(np.float32)  # type: ignore[union-attr]
        else:
            query_vec = None
        scores = _cosine_similarity_or_overlap(query, self._entries, self._matrix, query_vec)  # type: ignore[arg-type]
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results: list[tuple[ControlEntry, float]] = []
        for idx, score in ranked:
            entry = self._index[idx]
            if framework_filter and entry.framework.value != framework_filter:
                continue
            results.append((entry, float(score)))
            if len(results) >= top_k:
                break
        return results

    def crosswalk_by_similarity(
        self,
        source_entry: ControlEntry,
        target_framework: str,
        top_k: int = 3,
    ) -> list[tuple[ControlEntry, float]]:
        return self.search(
            query=source_entry.text_for_embedding(),
            top_k=top_k,
            framework_filter=target_framework,
        )


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _build_with_sentence_transformers(texts: Sequence[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(list(texts), show_progress_bar=True, convert_to_numpy=True)
    return embeddings.astype(np.float32)


def _build_with_tfidf(
    texts: Sequence[str],
) -> tuple[np.ndarray, object, dict[str, int]]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.85, min_df=1)
    matrix = vectorizer.fit_transform(list(texts)).toarray().astype(np.float32)
    vocabulary = dict(vectorizer.vocabulary_)
    return matrix, vectorizer, vocabulary


def _cosine_similarity_or_overlap(
    query: str,
    entries: list[ControlEntry],
    matrix: np.ndarray,
    query_vec: np.ndarray | None,
) -> np.ndarray:
    if query_vec is not None and query_vec.shape[1] == matrix.shape[1]:
        return _cosine(query_vec, matrix)

    # Fallback to word-overlap scoring
    query_tokens = set(query.lower().split())
    scores = np.zeros(matrix.shape[0], dtype=np.float32)
    for i, entry in enumerate(entries):
        entry_tokens = set(entry.text_for_embedding().lower().split())
        overlap = len(query_tokens & entry_tokens)
        union = len(query_tokens | entry_tokens)
        scores[i] = overlap / max(union, 1)
    return scores


def _cosine(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q_norm = np.linalg.norm(query_vec)
    m_norms = np.linalg.norm(matrix, axis=1)
    if q_norm == 0 or (m_norms == 0).any():
        return np.zeros(matrix.shape[0], dtype=np.float32)
    return np.dot(matrix, query_vec.T).flatten() / (m_norms * q_norm)
