# Local research retrieval (RAG Part A)

This package prepares local PDFs and text files for semantic search. It does not call Gemini, alter ChemBERTa predictions, or connect retrieved passages to the Q&A assistant. Retrieved passages include source metadata so a later integration can provide traceable context.

## Add documents

Create `research_documents/` at the repository root and place research-paper PDFs or UTF-8 `.txt` files there. Nested folders are supported. Keep these documents local; the directory is ignored by Git.

PDF text is extracted page by page with PyMuPDF. The PDF metadata title is preserved when present, and each result records filename, 1-based page number, and a `filename#page=N` source reference. Plain-text files produce one source record with no page number. Empty or scanned PDFs with no selectable text, corrupt files, and unreadable files are reported and skipped. A SHA-256 hash of normalized extracted text flags identical-content documents during a rebuild. Each source is still indexed with its own title, filename, and page references so provenance is retained.

Text cleaning removes NUL and soft-hyphen characters and normalizes whitespace. It preserves scientific symbols, punctuation, equations, and hyphenated terms. Each page or text document is split into overlapping word windows; every chunk keeps its source title, filename, page, reference, document hash, and chunk number.

## Build or rebuild

Install the repository requirements, then run from the repository root:

```powershell
py -3.10 -m pip install -r requirements.txt
New-Item -ItemType Directory -Force research_documents
py -3.10 -m rag.cli build --documents research_documents --index-dir rag_store
```

The default embedder is `sentence-transformers/all-MiniLM-L6-v2`. Sentence Transformers may download that embedding model on the first build if it is not already cached. No research documents are downloaded. To adjust chunk sizes:

```powershell
py -3.10 -m rag.cli build --chunk-words 220 --overlap-words 40
```

Adding documents and rebuilding replaces the local index with one built from all supported files currently in `research_documents/`. This intentionally simple full rebuild also removes entries for files that have been removed from the input folder.

## Search

```powershell
py -3.10 -m rag.cli search --question "What evidence discusses oral bioavailability?" --top-k 5
```

The question is embedded with the same model. Vectors are L2-normalized and searched with FAISS inner product, which is cosine similarity for normalized vectors. Results include similarity score, chunk text, title, filename, page number, and source reference. Search does not synthesize or paraphrase an answer.

## Stored index files

The default index directory is `rag_store/`, which is ignored by Git:

- `index.faiss`: FAISS `IndexFlatIP` vectors.
- `chunks.json`: chunk text and source metadata aligned by FAISS row number.
- `manifest.json`: format version, embedding model, dimensions, count, and metadata checksum.

All three files are required to load an index. Missing files produce a clear build-first message; corrupt metadata or incompatible model names are reported instead of returning mismatched sources. The generated index is local and can be recreated from the source documents.

## Tests

Run the RAG tests with:

```powershell
py -3.10 -m unittest tests.test_rag_pipeline -v
```

Tests mock PDF extraction, embeddings, and FAISS persistence. They make no model downloads, document downloads, or LLM/API calls.
