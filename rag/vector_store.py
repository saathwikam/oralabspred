from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .documents import DocumentChunk


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_FILENAME = "index.faiss"
CHUNKS_FILENAME = "chunks.json"
MANIFEST_FILENAME = "manifest.json"
INDEX_FORMAT_VERSION = 1


class RAGError(RuntimeError):
    """Base class for document retrieval errors."""


class EmbeddingDependencyError(RAGError):
    """Sentence Transformers is unavailable or failed to load the embedding model."""


class FaissDependencyError(RAGError):
    """The FAISS CPU package is unavailable."""


class EmptyQuestionError(ValueError):
    """A retrieval question is empty."""


class EmptyCorpusError(RAGError):
    """There are no text chunks to index."""


class IndexNotFoundError(RAGError):
    """One or more persistent index files are missing."""


class IndexCorruptError(RAGError):
    """The persisted index files are invalid or inconsistent."""


class EmbeddingModelMismatchError(RAGError):
    """The requested embedding model differs from the one used to build the index."""


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    filename: str
    title: str | None
    page_number: int | None
    source_reference: str
    score: float
    metadata: dict[str, Any]


def _load_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbeddingDependencyError(
            "Semantic retrieval requires sentence-transformers. Install the project requirements."
        ) from exc
    try:
        return SentenceTransformer(model_name, device="cpu")
    except Exception as exc:
        raise EmbeddingDependencyError(
            f"Could not load embedding model '{model_name}'. Check that it is available locally or that network access works."
        ) from exc


def _load_faiss():
    try:
        import faiss
    except ImportError as exc:
        raise FaissDependencyError(
            "FAISS support requires faiss-cpu. Install the project requirements."
        ) from exc
    return faiss


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ResearchIndex:
    """Cosine-similarity search over locally embedded research-document chunks."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        encoder: Any | None = None,
        faiss_module: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self._encoder = encoder
        self._faiss_module = faiss_module
        self._index: Any | None = None
        self._chunks: list[dict[str, Any]] = []

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def is_empty(self) -> bool:
        return self._index is None or self.chunk_count == 0

    def _get_encoder(self):
        if self._encoder is None:
            self._encoder = _load_sentence_transformer(self.model_name)
        return self._encoder

    def _get_faiss(self):
        if self._faiss_module is None:
            self._faiss_module = _load_faiss()
        return self._faiss_module

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            raise ValueError("At least one text is required to generate embeddings.")
        try:
            encoded = self._get_encoder().encode(
                list(texts),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except EmbeddingDependencyError:
            raise
        except Exception as exc:
            raise EmbeddingDependencyError(f"Embedding generation failed: {exc}") from exc
        vectors = np.asarray(encoded, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts) or vectors.shape[1] == 0:
            raise EmbeddingDependencyError("The embedding model returned an invalid vector shape.")
        if not np.isfinite(vectors).all():
            raise EmbeddingDependencyError("The embedding model returned non-finite values.")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise EmbeddingDependencyError("The embedding model returned a zero-length vector.")
        return np.ascontiguousarray(vectors / norms, dtype=np.float32)

    @staticmethod
    def _chunk_record(chunk: DocumentChunk | dict[str, Any]) -> dict[str, Any]:
        if isinstance(chunk, DocumentChunk):
            text, metadata = chunk.text, chunk.metadata
        else:
            text, metadata = chunk.get("text"), chunk.get("metadata")
        if not isinstance(text, str) or not text.strip() or not isinstance(metadata, dict):
            raise ValueError("Each indexed chunk must have non-empty text and a metadata object.")
        return {"text": text, "metadata": dict(metadata)}

    def build(self, chunks: Sequence[DocumentChunk | dict[str, Any]]) -> int:
        records = [self._chunk_record(chunk) for chunk in chunks]
        if not records:
            self._index = None
            self._chunks = []
            raise EmptyCorpusError("No document chunks are available to index.")

        vectors = self.embed_texts([record["text"] for record in records])
        try:
            index = self._get_faiss().IndexFlatIP(int(vectors.shape[1]))
            index.add(vectors)
        except FaissDependencyError:
            raise
        except Exception as exc:
            raise RAGError(f"Could not build the FAISS index: {exc}") from exc
        self._index = index
        self._chunks = records
        return len(records)

    def retrieve(self, question: str, top_k: int = 5) -> list[RetrievedChunk]:
        clean_question = question.strip() if isinstance(question, str) else ""
        if not clean_question:
            raise EmptyQuestionError("Enter a non-empty search question.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if self.is_empty:
            return []

        query = self.embed_texts([clean_question])
        if query.shape[1] != self._index.d:
            raise IndexCorruptError(
                "The configured embedding model dimension does not match the saved index."
            )
        try:
            scores, indices = self._index.search(query, min(top_k, self.chunk_count))
        except Exception as exc:
            raise RAGError(f"FAISS retrieval failed: {exc}") from exc

        results: list[RetrievedChunk] = []
        for score, row_index in zip(scores[0], indices[0], strict=True):
            if row_index < 0 or row_index >= self.chunk_count:
                continue
            record = self._chunks[int(row_index)]
            metadata = record["metadata"]
            results.append(
                RetrievedChunk(
                    text=record["text"],
                    filename=str(metadata.get("filename") or ""),
                    title=metadata.get("title"),
                    page_number=metadata.get("page_number"),
                    source_reference=str(metadata.get("source_reference") or ""),
                    score=float(score),
                    metadata=dict(metadata),
                )
            )
        return results

    def save(self, index_dir: str | Path) -> Path:
        if self.is_empty:
            raise EmptyCorpusError("Cannot save an empty research index.")
        directory = Path(index_dir).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        index_path = directory / INDEX_FILENAME
        chunks_path = directory / CHUNKS_FILENAME
        manifest_path = directory / MANIFEST_FILENAME
        temp_index = directory / f"{INDEX_FILENAME}.tmp"
        temp_chunks = directory / f"{CHUNKS_FILENAME}.tmp"
        temp_manifest = directory / f"{MANIFEST_FILENAME}.tmp"

        chunks_bytes = json.dumps(
            self._chunks, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        try:
            self._get_faiss().write_index(self._index, str(temp_index))
            temp_chunks.write_bytes(chunks_bytes)
            manifest = {
                "format_version": INDEX_FORMAT_VERSION,
                "embedding_model": self.model_name,
                "dimension": int(self._index.d),
                "chunk_count": self.chunk_count,
                "chunks_sha256": _sha256(chunks_bytes),
            }
            temp_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            os.replace(temp_index, index_path)
            os.replace(temp_chunks, chunks_path)
            os.replace(temp_manifest, manifest_path)
        except Exception as exc:
            for temp_path in (temp_index, temp_chunks, temp_manifest):
                temp_path.unlink(missing_ok=True)
            if isinstance(exc, FaissDependencyError):
                raise
            raise RAGError(f"Could not save the research index to '{directory}': {exc}") from exc
        return directory

    @classmethod
    def load(
        cls,
        index_dir: str | Path,
        *,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        encoder: Any | None = None,
        faiss_module: Any | None = None,
    ) -> "ResearchIndex":
        directory = Path(index_dir).expanduser()
        index_path = directory / INDEX_FILENAME
        chunks_path = directory / CHUNKS_FILENAME
        manifest_path = directory / MANIFEST_FILENAME
        if not all(path.is_file() for path in (index_path, chunks_path, manifest_path)):
            raise IndexNotFoundError(
                f"Research index is missing or incomplete in '{directory}'. Build it first."
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            chunks_bytes = chunks_path.read_bytes()
            if manifest.get("format_version") != INDEX_FORMAT_VERSION:
                raise IndexCorruptError("Unsupported research index format version.")
            stored_model = manifest.get("embedding_model")
            if stored_model != model_name:
                raise EmbeddingModelMismatchError(
                    f"Index uses '{stored_model}', but '{model_name}' was requested."
                )
            if _sha256(chunks_bytes) != manifest.get("chunks_sha256"):
                raise IndexCorruptError("Chunk metadata checksum does not match the manifest.")
            chunks = json.loads(chunks_bytes.decode("utf-8"))
            if not isinstance(chunks, list):
                raise IndexCorruptError("Chunk metadata is not a list.")
            for record in chunks:
                if (
                    not isinstance(record, dict)
                    or not isinstance(record.get("text"), str)
                    or not record["text"].strip()
                    or not isinstance(record.get("metadata"), dict)
                ):
                    raise IndexCorruptError("A stored research chunk is missing text or metadata.")
                metadata = record["metadata"]
                filename = metadata.get("filename")
                if not isinstance(filename, str) or not filename.strip():
                    raise IndexCorruptError("A stored research chunk is missing its source filename.")
                page_number = metadata.get("page_number")
                if filename.casefold().endswith(".pdf") and (
                    not isinstance(page_number, int) or page_number < 1
                ):
                    raise IndexCorruptError("A stored PDF chunk is missing its source page number.")
            resolved_faiss = faiss_module if faiss_module is not None else _load_faiss()
            index = resolved_faiss.read_index(str(index_path))
            if len(chunks) != manifest.get("chunk_count"):
                raise IndexCorruptError("Chunk metadata count does not match the manifest.")
            if len(chunks) != index.ntotal or index.d != manifest.get("dimension"):
                raise IndexCorruptError("FAISS index dimensions do not match the manifest.")
        except (IndexCorruptError, EmbeddingModelMismatchError, FaissDependencyError):
            raise
        except Exception as exc:
            raise IndexCorruptError(f"Could not load research index from '{directory}': {exc}") from exc

        result = cls(model_name=model_name, encoder=encoder, faiss_module=resolved_faiss)
        result._index = index
        result._chunks = chunks
        return result
