from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "p2_chemberta"))

from llm_assistant import GeminiNetworkError, GeminiQuotaError, GeminiServiceError
from qa_workflow import _structured_fallback_answer, answer_with_research
from rag.vector_store import IndexNotFoundError, RetrievedChunk


class FakeModels:
    def __init__(self, text="Evidence answer.", error=None):
        self.text = text
        self.error = error
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(text=self.text)


class FakeClient:
    def __init__(self, text="Evidence answer.", error=None):
        self.models = FakeModels(text=text, error=error)


class Unavailable503(Exception):
    status_code = 503
    status = "UNAVAILABLE"


def passage(text="The study reports oral absorption findings."):
    return RetrievedChunk(
        text=text,
        filename="absorption-study.pdf",
        title="Oral Absorption Study",
        page_number=7,
        source_reference="papers/absorption-study.pdf#page=7",
        score=0.84,
        metadata={
            "filename": "absorption-study.pdf",
            "title": "Oral Absorption Study",
            "page_number": 7,
            "source_reference": "papers/absorption-study.pdf#page=7",
        },
    )


class RagGeminiIntegrationTests(unittest.TestCase):
    def test_structured_fallback_uses_only_supplied_prediction_values(self):
        answer = _structured_fallback_answer(
            "Explain the HIA, HOB20 and HOB50 predictions",
            {
                "HIA": {"prediction": 1, "probability": 0.82},
                "HOB20": {"prediction": 0},
            },
            None,
            (),
        )
        self.assertIn("HIA is positive at 82.0%", answer)
        self.assertIn("HOB20 is classified negative", answer)
        self.assertIn("HOB50 is unavailable", answer)
        self.assertIn("not experimental measurements", answer)

    def test_structured_fallback_refuses_unsupported_custom_questions(self):
        self.assertEqual(
            _structured_fallback_answer(
                "What dose should I take?", None, None, ()
            ),
            "",
        )

    def test_gemini_failure_keeps_retrieved_evidence_for_structured_research_answer(self):
        index = SimpleNamespace(retrieve=Mock(return_value=[passage()]))
        client = FakeClient(error=RuntimeError("RESOURCE_EXHAUSTED quota exceeded"))
        with patch("qa_workflow.ResearchIndex.load", return_value=index):
            result = answer_with_research(
                "Curcumin",
                {},
                "What does the research report about oral bioavailability?",
                api_key="test-key",
                client=client,
                research_index=index,
                structured_fallback=True,
            )
        self.assertEqual(result.answer_mode, "structured")
        self.assertIn("cannot safely synthesize", result.answer)
        self.assertEqual(result.research_passages, (passage(),))

    def test_retrieved_context_and_sources_reach_gemini(self):
        client = FakeClient(text="The paper reports this result [absorption-study.pdf, p. 7].")
        index = SimpleNamespace(retrieve=Mock(return_value=[passage()]))
        compound_info = {"PubChem": {"CID": 2244}}
        predictions = {"HIA": {"prediction": 1, "probability": 0.91}}

        with patch("qa_workflow.ResearchIndex.load", return_value=index) as load_index:
            result = answer_with_research(
                "Aspirin",
                compound_info,
                "What does the paper report?",
                predictions,
                index_dir="custom-index",
                top_k=3,
                api_key="test-key",
                gemini_model="test-model",
                client=client,
            )

        load_index.assert_called_once()
        index.retrieve.assert_called_once_with("What does the paper report?", top_k=3)
        request = client.models.calls[0]
        prompt = request["contents"]
        self.assertEqual(request["model"], "test-model")
        self.assertIn("A. PubChem and ChEMBL compound information", prompt)
        self.assertIn("B. Research evidence retrieved through RAG", prompt)
        self.assertIn("C. Existing ChemBERTa predictions", prompt)
        self.assertIn("D. User's question", prompt)
        self.assertIn("absorption-study.pdf, p. 7", prompt)
        self.assertIn("papers/absorption-study.pdf#page=7", prompt)
        self.assertIn("The study reports oral absorption findings.", prompt)
        self.assertIn("0.91", prompt)
        self.assertEqual(result.research_passages, (passage(),))
        self.assertIn("[absorption-study.pdf, p. 7]", result.answer)

    def test_missing_index_still_answers_with_clear_evidence_limitation(self):
        client = FakeClient(text="The research documents available to me do not provide sufficient evidence.")
        with patch(
            "qa_workflow.ResearchIndex.load",
            side_effect=IndexNotFoundError("not found"),
        ):
            result = answer_with_research(
                "Aspirin",
                {"PubChem": {"CID": 2244}},
                "What do the studies show?",
                api_key="test-key",
                client=client,
            )

        self.assertEqual(result.research_passages, ())
        self.assertEqual(result.retrieval_error, "IndexNotFoundError")
        prompt = client.models.calls[0]["contents"]
        self.assertIn("Research retrieval was unavailable", prompt)
        self.assertIn("IndexNotFoundError", prompt)
        self.assertIn("do not provide sufficient evidence", prompt)
        self.assertIn('"CID": 2244', prompt)

    def test_empty_retrieval_is_explicitly_marked_as_insufficient(self):
        client = FakeClient()
        index = SimpleNamespace(retrieve=lambda _question, top_k: [])
        with patch("qa_workflow.ResearchIndex.load", return_value=index):
            result = answer_with_research(
                "",
                {},
                "Explain oral absorption generally.",
                api_key="test-key",
                client=client,
            )

        prompt = client.models.calls[0]["contents"]
        self.assertIsNone(result.retrieval_error)
        self.assertEqual(result.research_passages, ())
        self.assertIn("No relevant research passages were retrieved.", prompt)
        self.assertIn("General scientific explanations must be clearly labeled", prompt)

    def test_gemini_quota_failure_does_not_change_predictions(self):
        predictions = {
            "HIA": {"prediction": 1, "probability": 0.81},
            "HOB20": None,
            "HOB50": {"prediction": 0, "probability": 0.12},
        }
        original = {
            key: value.copy() if isinstance(value, dict) else value
            for key, value in predictions.items()
        }
        index = SimpleNamespace(retrieve=lambda _question, top_k: [passage()])
        client = FakeClient(error=RuntimeError("RESOURCE_EXHAUSTED quota exceeded"))

        with patch("qa_workflow.ResearchIndex.load", return_value=index):
            result = answer_with_research(
                "Aspirin",
                {},
                "Explain the HIA output.",
                predictions,
                api_key="test-key",
                client=client,
            )

        self.assertEqual(predictions, original)
        self.assertEqual(result.assistant_error_type, GeminiQuotaError.__name__)
        self.assertEqual(result.research_passages, (passage(),))
        prompt = client.models.calls[0]["contents"]
        self.assertIn('"HOB20": null', prompt)
        self.assertIn("never as negative", prompt)

    def test_network_failure_returns_retrieved_sources_and_preserves_predictions(self):
        class ConnectError(RuntimeError):
            pass

        predictions = {"HIA": {"prediction": 1, "probability": 0.81}}
        original = {key: value.copy() for key, value in predictions.items()}
        index = SimpleNamespace(retrieve=lambda _question, top_k: [passage()])
        client = FakeClient(error=ConnectError("DNS lookup failed"))

        with patch("qa_workflow.ResearchIndex.load", return_value=index):
            result = answer_with_research(
                "Curcumin",
                {},
                "What does the research say about oral bioavailability?",
                predictions,
                api_key="test-key",
                client=client,
            )

        self.assertEqual(result.answer, "")
        self.assertEqual(result.assistant_error_type, GeminiNetworkError.__name__)
        self.assertIn("network connection could not be established", result.assistant_error)
        self.assertEqual(result.research_passages, (passage(),))
        self.assertEqual(result.research_passages[0].filename, "absorption-study.pdf")
        self.assertEqual(result.research_passages[0].page_number, 7)
        self.assertEqual(predictions, original)

    def test_exhausted_503_retries_preserve_rag_sources_and_predictions(self):
        predictions = {"HIA": {"prediction": 1, "probability": 0.81}}
        original = {key: value.copy() for key, value in predictions.items()}
        index = SimpleNamespace(retrieve=Mock(return_value=[passage()]))
        client = FakeClient(error=Unavailable503("503 UNAVAILABLE"))

        with patch("qa_workflow.ResearchIndex.load", return_value=index), patch(
            "llm_assistant.time.sleep"
        ) as sleep:
            result = answer_with_research(
                "Curcumin",
                {},
                "What does the research say?",
                predictions,
                api_key="test-key",
                client=client,
            )

        self.assertEqual(len(client.models.calls), 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [2, 4])
        index.retrieve.assert_called_once_with("What does the research say?", top_k=5)
        self.assertEqual(result.answer, "")
        self.assertEqual(result.assistant_error_type, GeminiServiceError.__name__)
        self.assertIn("Gemini is temporarily unavailable", result.assistant_error)
        self.assertEqual(result.research_passages, (passage(),))
        self.assertEqual(result.research_passages[0].filename, "absorption-study.pdf")
        self.assertEqual(result.research_passages[0].page_number, 7)
        self.assertEqual(predictions, original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
