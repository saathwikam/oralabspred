from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .documents import (
    DocumentChunk,
    DocumentExtractionError,
    chunk_pages,
    discover_documents,
    extract_document,
)
from .vector_store import DEFAULT_EMBEDDING_MODEL, RAGError, ResearchIndex


class NoDocumentsFoundError(RAGError):
    """The input directory contains no supported documents."""


class NoUsableDocumentsError(RAGError):
    """No input document yielded extractable, indexable text."""


@dataclass(frozen=True)
class BuildReport:
    discovered_files: int
    indexed_files: int
    duplicate_files: tuple[str, ...]
    failed_files: tuple[tuple[str, str], ...]
    chunk_count: int
    index_directory: str


def build_corpus(
    documents_dir: str | Path = "research_documents",
    index_dir: str | Path = "rag_store",
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    chunk_words: int = 220,
    overlap_words: int = 40,
    encoder=None,
    faiss_module=None,
) -> BuildReport:
    """Extract, flag duplicate content, and index each source with its metadata."""
    paths = discover_documents(documents_dir)
    if not paths:
        raise NoDocumentsFoundError(
            f"No PDF or TXT documents were found in '{documents_dir}'."
        )

    chunks: list[DocumentChunk] = []
    failed: list[tuple[str, str]] = []
    duplicates: list[str] = []
    seen_document_hashes: set[str] = set()
    indexed_files = 0

    for path in paths:
        try:
            pages = extract_document(path, documents_root=documents_dir)
        except DocumentExtractionError as exc:
            failed.append((path.name, str(exc)))
            continue
        document_hash = pages[0].metadata["document_hash"]
        if document_hash in seen_document_hashes:
            # Identical content can still have distinct titles, filenames, or provenance.
            # Flag the duplicate, but retain its chunks so retrieval remains traceable.
            duplicates.append(path.name)
        else:
            seen_document_hashes.add(document_hash)
        file_chunks = chunk_pages(
            pages, chunk_words=chunk_words, overlap_words=overlap_words
        )
        if file_chunks:
            chunks.extend(file_chunks)
            indexed_files += 1

    if not chunks:
        detail = "; ".join(f"{name}: {message}" for name, message in failed)
        suffix = f" Details: {detail}" if detail else ""
        raise NoUsableDocumentsError(
            f"No usable text was found in {len(paths)} supported document(s).{suffix}"
        )

    index = ResearchIndex(
        model_name=model_name, encoder=encoder, faiss_module=faiss_module
    )
    chunk_count = index.build(chunks)
    index.save(index_dir)
    return BuildReport(
        discovered_files=len(paths),
        indexed_files=indexed_files,
        duplicate_files=tuple(duplicates),
        failed_files=tuple(failed),
        chunk_count=chunk_count,
        index_directory=str(Path(index_dir).expanduser()),
    )
