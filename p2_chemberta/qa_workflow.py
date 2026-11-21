from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.vector_store import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    RAGError,
    ResearchIndex,
    RetrievedChunk,
)
from llm_assistant import GeminiAssistantError, answer_compound_question  # noqa: E402


DEFAULT_RAG_INDEX_DIR = PROJECT_ROOT / "rag_store"


@dataclass(frozen=True)
class CompoundQAResult:
    answer: str
    research_passages: tuple[RetrievedChunk, ...]
    retrieval_error: str | None = None
    assistant_error: str | None = None
    assistant_error_type: str | None = None
    answer_mode: str | None = None


def _structured_fallback_answer(
    question: str,
    predictions: Mapping[str, Any] | None,
    descriptors: Mapping[str, Any] | None,
    passages: tuple[RetrievedChunk, ...],
) -> str:
    """Answer only supported prediction/property questions without an LLM."""
    clean = question.casefold()
    endpoints = predictions or {}
    asks_predictions = any(token in clean for token in (
        "hia", "hob20", "hob50", "prediction", "probability", "classification", "model output"
    ))
    asks_properties = any(token in clean for token in (
        "molecular propert", "molecular weight", "logp", "tpsa", "polarity",
        "hydrogen bond", "descriptor", "smiles", "structure"
    ))
    asks_research = any(token in clean for token in (
        "research", "paper", "literature", "study", "studies", "formulation", "reported"
    ))

    if asks_predictions and endpoints:
        sentences = []
        for endpoint in ("HIA", "HOB20", "HOB50"):
            value = endpoints.get(endpoint)
            if not isinstance(value, Mapping) or value.get("prediction") not in (0, 1):
                sentences.append(f"{endpoint} is unavailable.")
                continue
            label = "positive" if int(value["prediction"]) == 1 else "negative"
            probability = value.get("probability")
            if isinstance(probability, (int, float)):
                sentences.append(f"{endpoint} is {label} at {float(probability) * 100:.1f}% predicted probability.")
            else:
                sentences.append(f"{endpoint} is classified {label}; a probability was not supplied.")
        sentences.append(
            "HOB20 and HOB50 are separate learned endpoint labels; these are computational predictions, not experimental measurements."
        )
        return " ".join(sentences)

    if asks_properties and descriptors:
        available = [
            f"{name} {value}"
            for name, value in descriptors.items()
            if isinstance(value, (int, float))
        ]
        if available:
            property_text = ", ".join(available)
            return (
                f"RDKit calculated these molecular descriptors from the submitted structure: {property_text}. "
                "These computed properties can inform general discussion of solubility and permeability, "
                "but they are not experimental absorption measurements and do not determine a compound's clinical performance."
            )

    if asks_research:
        if passages:
            source_count = len({passage.filename for passage in passages})
            return (
                "Gemini is unavailable, so I cannot safely synthesize the retrieved papers. "
                f"The local search found relevant passages from {source_count} research PDF(s); "
                "review the excerpts and page links below for the evidence."
            )
        return "No relevant research passages were retrieved, so there is not enough local evidence to answer this literature question."

    if any(token in clean for token in ("oral absorption", "bioavailability", "intestinal absorption", "permeability", "solubility")):
        return (
            "As general scientific context, oral absorption depends on interacting factors such as dissolution, "
            "permeability, molecular properties and transport or metabolism. No compound-specific experimental "
            "conclusion can be drawn from this structured fallback alone."
        )
    return ""


def answer_with_research(
    compound_name: str,
    compound_information: Mapping[str, Any] | None,
    question: str,
    chemberta_predictions: Mapping[str, Any] | None = None,
    *,
    index_dir: str | Path | None = None,
    top_k: int = 5,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    api_key: str | None = None,
    gemini_model: str | None = None,
    client: Any | None = None,
    research_index: ResearchIndex | None = None,
    research_index_error: str | None = None,
    molecular_descriptors: Mapping[str, Any] | None = None,
    assistant_instructions: str | None = None,
    structured_fallback: bool = False,
) -> CompoundQAResult:
    """Retrieve local research context, then pass it with records to Gemini.

    Retrieval failures are isolated from Gemini: the assistant still receives
    compound records and is told that no research evidence was available.
    """
    clean_question = question.strip() if isinstance(question, str) else ""
    if not clean_question:
        raise ValueError("Enter a question for the compound assistant.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    resolved_index_dir = (
        Path(index_dir).expanduser()
        if index_dir is not None
        else Path(os.environ.get("ORALABS_RAG_INDEX_DIR", DEFAULT_RAG_INDEX_DIR)).expanduser()
    )
    passages: tuple[RetrievedChunk, ...] = ()
    retrieval_error = research_index_error
    index = research_index
    if index is None and retrieval_error is None:
        try:
            index = ResearchIndex.load(
                resolved_index_dir,
                model_name=embedding_model,
            )
        except (RAGError, OSError) as exc:
            # Expose a stable, non-path-bearing status to the model and UI.
            retrieval_error = type(exc).__name__
    if index is not None:
        try:
            passages = tuple(index.retrieve(clean_question, top_k=top_k))
        except (RAGError, OSError) as exc:
            retrieval_error = type(exc).__name__

    try:
        answer = answer_compound_question(
            compound_name=compound_name,
            compound_information=compound_information,
            question=clean_question,
            chemberta_predictions=chemberta_predictions,
            research_passages=passages,
            research_availability_error=retrieval_error,
            assistant_instructions=assistant_instructions,
            api_key=api_key,
            model_name=gemini_model,
            client=client,
        )
    except GeminiAssistantError as exc:
        # Return retrieved passages even when Gemini is unavailable, so the UI
        # can preserve source context and report the assistant error separately.
        fallback_answer = (
            _structured_fallback_answer(
                clean_question,
                chemberta_predictions,
                molecular_descriptors,
                passages,
            )
            if structured_fallback
            else ""
        )
        return CompoundQAResult(
            answer=fallback_answer,
            research_passages=passages,
            retrieval_error=retrieval_error,
            assistant_error=str(exc),
            assistant_error_type=type(exc).__name__,
            answer_mode="structured" if fallback_answer else None,
        )
    return CompoundQAResult(
        answer=answer,
        research_passages=passages,
        retrieval_error=retrieval_error,
        answer_mode="gemini",
    )
