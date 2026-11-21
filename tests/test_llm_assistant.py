from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "p2_chemberta"))

from llm_assistant import (  # noqa: E402
    DEFAULT_GEMINI_MODEL,
    GeminiAPIError,
    GeminiConfigurationError,
    GeminiQuotaError,
    GeminiServiceError,
    GeminiTimeoutError,
    answer_compound_question,
    get_gemini_api_key,
)


class FakeModels:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, response=None, error=None):
        self.models = FakeModels(response=response, error=error)


class SequencedModels:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class Unavailable503(Exception):
    status_code = 503
    status = "UNAVAILABLE"


class Deadline504(Exception):
    status_code = 504
    status = "DEADLINE_EXCEEDED"


class GeminiAssistantTests(unittest.TestCase):
    def test_successful_first_request_is_not_retried(self) -> None:
        client = FakeClient(response=SimpleNamespace(text="Answer."))
        with patch("llm_assistant.time.sleep") as sleep:
            answer = answer_compound_question(
                "Aspirin", {}, "Question?", api_key="test-key", client=client
            )
        self.assertEqual(answer, "Answer.")
        self.assertEqual(len(client.models.calls), 1)
        sleep.assert_not_called()

    def test_503_retries_after_two_and_four_seconds_then_returns_answer(self) -> None:
        models = SequencedModels([
            Unavailable503("503 UNAVAILABLE"),
            Unavailable503("503 UNAVAILABLE"),
            SimpleNamespace(text="Recovered answer."),
        ])
        client = SimpleNamespace(models=models)
        with patch("llm_assistant.time.sleep") as sleep:
            answer = answer_compound_question(
                "Aspirin", {}, "Question?", api_key="test-key", client=client
            )
        self.assertEqual(answer, "Recovered answer.")
        self.assertEqual(len(models.calls), 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [2, 4])

    def test_504_deadline_exceeded_retries_then_returns_answer(self) -> None:
        models = SequencedModels([Deadline504("504 DEADLINE_EXCEEDED"), SimpleNamespace(text="Recovered.")])
        with patch("llm_assistant.time.sleep") as sleep:
            answer = answer_compound_question(
                "Aspirin", {}, "Question?", api_key="test-key",
                client=SimpleNamespace(models=models),
            )
        self.assertEqual(answer, "Recovered.")
        self.assertEqual(len(models.calls), 2)
        sleep.assert_called_once_with(2)

    def test_three_504_attempts_return_friendly_error(self) -> None:
        models = SequencedModels([Deadline504("504 DEADLINE_EXCEEDED") for _ in range(3)])
        with patch("llm_assistant.time.sleep") as sleep:
            with self.assertRaises(GeminiServiceError) as context:
                answer_compound_question(
                    "Aspirin", {}, "Question?", api_key="test-key",
                    client=SimpleNamespace(models=models),
                )
        self.assertEqual(len(models.calls), 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [2, 4])
        self.assertIn("temporarily unavailable", str(context.exception))

    def test_three_503_attempts_return_friendly_error(self) -> None:
        models = SequencedModels([Unavailable503("503 UNAVAILABLE") for _ in range(3)])
        with patch("llm_assistant.time.sleep") as sleep:
            with self.assertRaises(GeminiServiceError) as context:
                answer_compound_question(
                    "Aspirin", {}, "Question?", api_key="test-key",
                    client=SimpleNamespace(models=models),
                )
        self.assertEqual(len(models.calls), 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [2, 4])
        self.assertIn("temporarily unavailable", str(context.exception))
        self.assertIn("try again later", str(context.exception))

    def test_non_503_api_error_is_not_retried(self) -> None:
        class InvalidRequest(Exception):
            status_code = 400
            status = "INVALID_ARGUMENT"

        models = SequencedModels([InvalidRequest("bad request")])
        with patch("llm_assistant.time.sleep") as sleep:
            with self.assertRaises(GeminiAPIError):
                answer_compound_question(
                    "Aspirin", {}, "Question?", api_key="test-key",
                    client=SimpleNamespace(models=models),
                )
        self.assertEqual(len(models.calls), 1)
        sleep.assert_not_called()

    def test_api_key_resolution_prefers_environment_then_streamlit_secrets(self) -> None:
        with patch.dict(os.environ, {"GEMINI_API_KEY": " env-key "}):
            self.assertEqual(
                get_gemini_api_key({"GEMINI_API_KEY": "secret-key"}), "env-key"
            )
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                get_gemini_api_key({"gemini": {"api_key": "secret-key"}}), "secret-key"
            )

    def test_success_uses_configured_model_and_controlled_context(self) -> None:
        client = FakeClient(response=SimpleNamespace(text="Aspirin is an NSAID."))
        information = {
            "PubChem": {"CID": 2244, "record URL": "https://example.test/pubchem/2244"},
            "ChEMBL": {"molecule ID": "CHEMBL25", "molecular weight": 180.16},
        }
        predictions = {"HIA": {"probability": 0.82, "prediction": 1}}

        answer = answer_compound_question(
            "Aspirin",
            information,
            "What is Aspirin?",
            predictions,
            api_key="test-key",
            model_name="test-model",
            client=client,
        )

        self.assertEqual(answer, "Aspirin is an NSAID.")
        request = client.models.calls[0]
        self.assertEqual(request["model"], "test-model")
        self.assertEqual(request["config"], {"max_output_tokens": 1024})
        prompt = request["contents"]
        self.assertIn("CHEMBL25", prompt)
        self.assertIn("0.82", prompt)
        self.assertIn("never", prompt.lower())
        self.assertIn("Do not fabricate citations", prompt)

    def test_missing_key_is_reported_without_api_request(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(GeminiConfigurationError):
                answer_compound_question("Aspirin", {}, "What is it?")

    def test_explicit_empty_key_does_not_reread_environment(self) -> None:
        with patch.dict(os.environ, {"GEMINI_API_KEY": "stale-environment-key"}):
            with self.assertRaises(GeminiConfigurationError):
                answer_compound_question(
                    "Aspirin", {}, "What is it?", api_key=""
                )

    def test_quota_error_is_classified(self) -> None:
        class QuotaFailure(RuntimeError):
            code = 429

        client = FakeClient(error=QuotaFailure("RESOURCE_EXHAUSTED"))
        with self.assertRaises(GeminiQuotaError):
            answer_compound_question(
                "Aspirin", {}, "What is it?", api_key="test-key", client=client
            )

    def test_network_and_api_failures_are_safely_reported(self) -> None:
        client = FakeClient(error=ConnectionError("network unreachable"))
        with self.assertRaises(GeminiServiceError):
            answer_compound_question(
                "Aspirin", {}, "What is it?", api_key="test-key", client=client
            )

    def test_timeout_and_api_failures_are_classified_without_raw_message(self) -> None:
        class ReadTimeout(Exception):
            pass

        with self.assertRaises(GeminiTimeoutError):
            answer_compound_question(
                "Aspirin",
                {},
                "What is it?",
                api_key="test-key",
                client=FakeClient(error=ReadTimeout("request timed out")),
            )

        class APIError(Exception):
            status_code = 400
            status = "INVALID_ARGUMENT"

        with self.assertRaises(GeminiAPIError) as context:
            answer_compound_question(
                "Aspirin",
                {},
                "What is it?",
                api_key="test-key",
                client=FakeClient(error=APIError("private raw detail")),
            )
        self.assertIn("HTTP/status 400", str(context.exception))
        self.assertIn("INVALID_ARGUMENT", str(context.exception))
        self.assertNotIn("private raw detail", str(context.exception))

    def test_empty_response_is_reported(self) -> None:
        client = FakeClient(response=SimpleNamespace(text="  "))
        with self.assertRaises(GeminiServiceError):
            answer_compound_question(
                "Aspirin", {}, "What is it?", api_key="test-key", client=client
            )

    def test_model_name_has_configurable_default(self) -> None:
        client = FakeClient(response=SimpleNamespace(text="Answer."))
        with patch.dict(os.environ, {"GEMINI_MODEL": "configured-model"}):
            answer_compound_question(
                "Aspirin", {}, "What is it?", api_key="test-key", client=client
            )
        self.assertEqual(client.models.calls[0]["model"], "configured-model")
        self.assertEqual(DEFAULT_GEMINI_MODEL, "gemini-3.8-flash")


if __name__ == "__main__":
    unittest.main(verbosity=2)
