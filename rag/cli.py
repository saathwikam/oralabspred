from __future__ import annotations

import argparse
import sys

from .pipeline import build_corpus
from .vector_store import (
    DEFAULT_EMBEDDING_MODEL,
    IndexNotFoundError,
    ResearchIndex,
    RAGError,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and search the local Oralabs Predict research index."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="Rebuild the index from PDF and TXT files.")
    build.add_argument("--documents", default="research_documents")
    build.add_argument("--index-dir", default="rag_store")
    build.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    build.add_argument("--chunk-words", type=int, default=220)
    build.add_argument("--overlap-words", type=int, default=40)

    search = commands.add_parser("search", help="Retrieve relevant indexed text chunks.")
    search.add_argument("--question", required=True)
    search.add_argument("--index-dir", default="rag_store")
    search.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    search.add_argument("--top-k", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build":
            report = build_corpus(
                documents_dir=args.documents,
                index_dir=args.index_dir,
                model_name=args.model,
                chunk_words=args.chunk_words,
                overlap_words=args.overlap_words,
            )
            print(
                f"Indexed {report.indexed_files} file(s) into {report.chunk_count} chunk(s) "
                f"at '{report.index_directory}'."
            )
            if report.duplicate_files:
                print("Identical-content files detected (sources retained): " + ", ".join(report.duplicate_files))
            for filename, error in report.failed_files:
                print(f"Skipped {filename}: {error}", file=sys.stderr)
            return 0

        index = ResearchIndex.load(args.index_dir, model_name=args.model)
        results = index.retrieve(args.question, top_k=args.top_k)
        if not results:
            print("The research index is empty; add documents and rebuild it.")
            return 0
        for rank, result in enumerate(results, start=1):
            title = result.title or result.filename
            print(f"[{rank}] {title} | {result.source_reference} | score={result.score:.4f}")
            print(result.text)
            print()
        return 0
    except IndexNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (RAGError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
