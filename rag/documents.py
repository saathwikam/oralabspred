from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUPPORTED_SUFFIXES = {".pdf", ".txt"}


class DocumentExtractionError(RuntimeError):
    """A document could not be read or contained no extractable text."""


class DocumentDependencyError(DocumentExtractionError):
    """A document reader dependency is unavailable."""


class UnsupportedDocumentError(DocumentExtractionError):
    """The input file type is not supported."""


class NoExtractableTextError(DocumentExtractionError):
    """A document has no selectable text, as with scanned image-only PDFs."""


@dataclass(frozen=True)
class DocumentPage:
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DocumentChunk:
    text: str
    metadata: dict[str, Any]


def clean_text(text: str) -> str:
    """Normalize extraction whitespace while retaining scientific characters."""
    text = text.replace("\x00", "").replace("\u00ad", "")
    lines = [re.sub(r"[\t\f\v ]+", " ", line).strip() for line in text.splitlines()]
    cleaned = "\n".join(line for line in lines if line)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _load_pymupdf():
    try:
        import pymupdf
    except ImportError as exc:
        raise DocumentDependencyError(
            "PDF support requires PyMuPDF. Install the project requirements."
        ) from exc
    return pymupdf


def _source_reference(path: Path, documents_root: str | Path | None) -> str:
    if documents_root is None:
        return path.name
    root = Path(documents_root).expanduser().resolve()
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.name


def _document_hash(texts: list[str]) -> str:
    normalized = "\n".join(clean_text(text) for text in texts)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _extract_pdf(path: Path, documents_root: str | Path | None) -> list[DocumentPage]:
    pymupdf = _load_pymupdf()
    try:
        document = pymupdf.open(str(path))
    except Exception as exc:
        raise DocumentExtractionError(f"Could not open PDF '{path.name}': {exc}") from exc

    pages: list[tuple[int, str]] = []
    try:
        document_metadata = getattr(document, "metadata", None) or {}
        raw_title = document_metadata.get("title") if isinstance(document_metadata, dict) else None
        title = raw_title.strip() if isinstance(raw_title, str) and raw_title.strip() else None
        for page_number, page in enumerate(document, start=1):
            try:
                page_text = clean_text(page.get_text("text") or "")
            except Exception as exc:
                raise DocumentExtractionError(
                    f"Could not extract page {page_number} from '{path.name}': {exc}"
                ) from exc
            if page_text:
                pages.append((page_number, page_text))
    finally:
        close = getattr(document, "close", None)
        if callable(close):
            close()

    if not pages:
        raise NoExtractableTextError(
            f"PDF '{path.name}' has no extractable text; it may be scanned or empty."
        )

    source = _source_reference(path, documents_root)
    content_hash = _document_hash([text for _, text in pages])
    return [
        DocumentPage(
            text=text,
            metadata={
                "title": title,
                "filename": path.name,
                "page_number": page_number,
                "source_reference": f"{source}#page={page_number}",
                "document_hash": content_hash,
            },
        )
        for page_number, text in pages
    ]


def _extract_text(path: Path, documents_root: str | Path | None) -> list[DocumentPage]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise DocumentExtractionError(f"Could not read '{path.name}': {exc}") from exc
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    text = clean_text(text)
    if not text:
        raise NoExtractableTextError(f"Text file '{path.name}' is empty.")

    return [
        DocumentPage(
            text=text,
            metadata={
                "title": None,
                "filename": path.name,
                "page_number": None,
                "source_reference": _source_reference(path, documents_root),
                "document_hash": _document_hash([text]),
            },
        )
    ]


def extract_document(
    path: str | Path, documents_root: str | Path | None = None
) -> list[DocumentPage]:
    """Extract PDF pages or one plain-text record with source metadata."""
    document_path = Path(path).expanduser()
    if not document_path.is_file():
        raise DocumentExtractionError(f"Document does not exist or is not a file: {document_path}")
    suffix = document_path.suffix.lower()
    if suffix == ".pdf":
        try:
            return _extract_pdf(document_path, documents_root)
        except DocumentExtractionError:
            raise
        except Exception as exc:
            raise DocumentExtractionError(f"Could not read PDF '{document_path.name}': {exc}") from exc
    if suffix == ".txt":
        return _extract_text(document_path, documents_root)
    raise UnsupportedDocumentError(
        f"Unsupported file type '{suffix or '(no extension)'}' for '{document_path.name}'."
    )


def discover_documents(documents_dir: str | Path) -> list[Path]:
    root = Path(documents_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Research document directory does not exist: {root}")
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES),
        key=lambda path: path.as_posix().lower(),
    )


def chunk_pages(
    pages: list[DocumentPage], *, chunk_words: int = 220, overlap_words: int = 40
) -> list[DocumentChunk]:
    """Split each extracted page into overlapping word windows, preserving metadata."""
    if chunk_words <= 0:
        raise ValueError("chunk_words must be greater than zero.")
    if overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("overlap_words must be non-negative and smaller than chunk_words.")

    chunks: list[DocumentChunk] = []
    chunk_index = 0
    for page in pages:
        words = page.text.split()
        start = 0
        while start < len(words):
            end = min(start + chunk_words, len(words))
            chunk_text = " ".join(words[start:end])
            if chunk_text:
                metadata = dict(page.metadata)
                metadata["chunk_index"] = chunk_index
                chunks.append(DocumentChunk(text=chunk_text, metadata=metadata))
                chunk_index += 1
            if end == len(words):
                break
            start = end - overlap_words
    return chunks
