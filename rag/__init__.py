"""Local research-document preparation and semantic retrieval."""

from .documents import (
    DocumentChunk,
    DocumentExtractionError,
    DocumentPage,
    chunk_pages,
    discover_documents,
    extract_document,
)
from .pipeline import BuildReport, build_corpus
from .vector_store import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingDependencyError,
    EmptyQuestionError,
    IndexNotFoundError,
    ResearchIndex,
    RetrievedChunk,
)

__all__ = [
    "BuildReport",
    "DEFAULT_EMBEDDING_MODEL",
    "DocumentChunk",
    "DocumentExtractionError",
    "DocumentPage",
    "EmbeddingDependencyError",
    "EmptyQuestionError",
    "IndexNotFoundError",
    "ResearchIndex",
    "RetrievedChunk",
    "build_corpus",
    "chunk_pages",
    "discover_documents",
    "extract_document",
]
