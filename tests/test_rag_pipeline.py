from __future__ import annotations

import json
import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from rag.documents import (
    DocumentChunk,
    DocumentDependencyError,
    DocumentExtractionError,
    DocumentPage,
    NoExtractableTextError,
    chunk_pages,
    extract_document,
)
from rag.pipeline import NoDocumentsFoundError, build_corpus
from rag.vector_store import (
    EmbeddingDependencyError,
    EmptyQuestionError,
    FaissDependencyError,
    IndexNotFoundError,
    ResearchIndex,
)


class FakePage:
    def __init__(self, text):
        self.text = text

    def get_text(self, _mode):
        return self.text


class FakePDF:
    metadata = {"title": "Absorption study"}

    def __init__(self, pages):
        self.pages = [FakePage(text) for text in pages]
        self.closed = False

    def __iter__(self):
        return iter(self.pages)

    def close(self):
        self.closed = True


class FakeEncoder:
    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            lowered = text.lower()
            # Stable two-dimensional semantic proxy for deterministic retrieval.
            vectors.append([1.0, 0.0] if "oral absorption" in lowered else [0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


class FakeFaissIndex:
    def __init__(self, dimension):
        self.d = dimension
        self.vectors = np.empty((0, dimension), dtype=np.float32)

    @property
    def ntotal(self):
        return int(self.vectors.shape[0])

    def add(self, vectors):
        self.vectors = np.vstack([self.vectors, vectors])

    def search(self, queries, count):
        scores = queries @ self.vectors.T
        order = np.argsort(-scores, axis=1)[:, :count]
        return np.take_along_axis(scores, order, axis=1), order.astype(np.int64)


class FakeFaiss:
    IndexFlatIP = FakeFaissIndex

    @staticmethod
    def write_index(index, path):
        with open(path, "wb") as output:
            pickle.dump((index.d, index.vectors), output)

    @staticmethod
    def read_index(path):
        with open(path, "rb") as source:
            dimension, vectors = pickle.load(source)
        index = FakeFaissIndex(dimension)
        index.vectors = vectors
        return index


class DocumentTests(unittest.TestCase):
    def test_text_extraction_cleans_whitespace_and_preserves_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file_path = root / "study.txt"
            file_path.write_text("  HIA\t≥ 30%  \n\nCYP3A4–mediated uptake.", encoding="utf-8")
            page = extract_document(file_path, documents_root=root)[0]

        self.assertEqual(page.text, "HIA ≥ 30%\nCYP3A4–mediated uptake.")
        self.assertEqual(page.metadata["filename"], "study.txt")
        self.assertIsNone(page.metadata["title"])
        self.assertIsNone(page.metadata["page_number"])
        self.assertEqual(page.metadata["source_reference"], "study.txt")
        self.assertEqual(len(page.metadata["document_hash"]), 64)

    def test_pdf_extraction_preserves_title_page_and_source(self):
        pdf = FakePDF(["", "Oral absorption evidence on page two."])
        fake_module = types.SimpleNamespace(open=lambda _path: pdf)
        with tempfile.TemporaryDirectory() as directory:
            file_path = Path(directory) / "paper.pdf"
            file_path.write_bytes(b"mock")
            with patch.dict(sys.modules, {"pymupdf": fake_module}):
                pages = extract_document(file_path, documents_root=directory)

        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0].metadata["title"], "Absorption study")
        self.assertEqual(pages[0].metadata["filename"], "paper.pdf")
        self.assertEqual(pages[0].metadata["page_number"], 2)
        self.assertEqual(pages[0].metadata["source_reference"], "paper.pdf#page=2")
        self.assertTrue(pdf.closed)

    def test_empty_scanned_pdf_and_corrupt_pdf_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            file_path = Path(directory) / "scan.pdf"
            file_path.write_bytes(b"mock")
            with patch.dict(sys.modules, {"pymupdf": types.SimpleNamespace(open=lambda _path: FakePDF(["", "  "]))}):
                with self.assertRaises(NoExtractableTextError):
                    extract_document(file_path)
            with patch.dict(sys.modules, {"pymupdf": types.SimpleNamespace(open=lambda _path: (_ for _ in ()).throw(ValueError("broken")))}):
                with self.assertRaises(DocumentExtractionError):
                    extract_document(file_path)

    def test_missing_pdf_dependency_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            file_path = Path(directory) / "paper.pdf"
            file_path.write_bytes(b"mock")
            with patch.dict(sys.modules, {"pymupdf": None}):
                with self.assertRaises(DocumentDependencyError):
                    extract_document(file_path)

    def test_chunking_overlaps_and_preserves_page_metadata(self):
        page = DocumentPage(
            "one two three four five six",
            {"filename": "p.pdf", "title": "Paper", "page_number": 4, "source_reference": "p.pdf#page=4"},
        )
        chunks = chunk_pages([page], chunk_words=4, overlap_words=2)

        self.assertEqual([chunk.text for chunk in chunks], ["one two three four", "three four five six"])
        self.assertEqual(chunks[0].metadata["page_number"], 4)
        self.assertEqual(chunks[0].metadata["source_reference"], "p.pdf#page=4")
        self.assertEqual([chunk.metadata["chunk_index"] for chunk in chunks], [0, 1])


class RetrievalTests(unittest.TestCase):
    def make_index(self):
        return ResearchIndex(encoder=FakeEncoder(), faiss_module=FakeFaiss)

    def test_embeddings_are_normalized_and_index_retrieves_metadata(self):
        index = self.make_index()
        vectors = index.embed_texts(["oral absorption", "liver toxicity"])
        self.assertEqual(vectors.shape, (2, 2))
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0])

        chunks = [
            DocumentChunk("Evidence about oral absorption", {"filename": "a.pdf", "title": "Study A", "page_number": 3, "source_reference": "a.pdf#page=3"}),
            DocumentChunk("Evidence about liver toxicity", {"filename": "b.pdf", "title": "Study B", "page_number": 8, "source_reference": "b.pdf#page=8"}),
        ]
        index.build(chunks)
        results = index.retrieve("oral absorption", top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].filename, "a.pdf")
        self.assertEqual(results[0].title, "Study A")
        self.assertEqual(results[0].page_number, 3)
        self.assertEqual(results[0].source_reference, "a.pdf#page=3")
        self.assertEqual(results[0].text, chunks[0].text)

    def test_index_persists_reloads_and_retrieves(self):
        index = self.make_index()
        index.build([
            DocumentChunk("Oral absorption evidence", {"filename": "paper.txt", "title": "Text study", "page_number": None, "source_reference": "paper.txt"})
        ])
        with tempfile.TemporaryDirectory() as directory:
            index.save(directory)
            self.assertTrue((Path(directory) / "index.faiss").is_file())
            self.assertTrue((Path(directory) / "chunks.json").is_file())
            self.assertTrue((Path(directory) / "manifest.json").is_file())
            loaded = ResearchIndex.load(directory, encoder=FakeEncoder(), faiss_module=FakeFaiss)
            result = loaded.retrieve("oral absorption")[0]

        self.assertEqual(loaded.chunk_count, 1)
        self.assertEqual(result.filename, "paper.txt")
        self.assertEqual(result.source_reference, "paper.txt")

    def test_missing_index_and_empty_question_are_clear(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(IndexNotFoundError):
                ResearchIndex.load(directory, faiss_module=FakeFaiss)
        index = self.make_index()
        self.assertEqual(index.retrieve("not indexed yet"), [])
        with self.assertRaises(EmptyQuestionError):
            index.retrieve("  ")

    def test_missing_embedding_dependency_is_clear(self):
        index = ResearchIndex()
        with patch("rag.vector_store._load_sentence_transformer", side_effect=ImportError("missing")):
            with self.assertRaises(EmbeddingDependencyError):
                index.embed_texts(["question"])

    def test_missing_faiss_dependency_is_clear(self):
        index = ResearchIndex(encoder=FakeEncoder())
        chunk = DocumentChunk("oral absorption", {"filename": "x.txt"})
        with patch("rag.vector_store._load_faiss", side_effect=FaissDependencyError("missing")):
            with self.assertRaises(FaissDependencyError):
                index.build([chunk])


class BuildPipelineTests(unittest.TestCase):
    def test_build_flags_duplicates_and_preserves_each_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "documents"
            root.mkdir()
            (root / "a.txt").write_text("Oral absorption research evidence.", encoding="utf-8")
            (root / "b.txt").write_text("Oral absorption research evidence.", encoding="utf-8")
            store = Path(directory) / "store"
            report = build_corpus(
                root,
                store,
                encoder=FakeEncoder(),
                faiss_module=FakeFaiss,
                chunk_words=20,
                overlap_words=2,
            )
            self.assertEqual(report.discovered_files, 2)
            self.assertEqual(report.indexed_files, 2)
            self.assertEqual(report.duplicate_files, ("b.txt",))
            self.assertEqual(report.chunk_count, 2)
            loaded = ResearchIndex.load(
                store, encoder=FakeEncoder(), faiss_module=FakeFaiss
            )
            hits = loaded.retrieve("oral absorption", top_k=2)
            self.assertEqual({hit.filename for hit in hits}, {"a.txt", "b.txt"})
            self.assertEqual({hit.source_reference for hit in hits}, {"a.txt", "b.txt"})

    def test_empty_corpus_is_not_silently_indexed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "documents"
            root.mkdir()
            with self.assertRaises(NoDocumentsFoundError):
                build_corpus(root, Path(directory) / "store", encoder=FakeEncoder(), faiss_module=FakeFaiss)


if __name__ == "__main__":
    unittest.main()
