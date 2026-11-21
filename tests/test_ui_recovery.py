from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "p2_chemberta"))

import ui_pages  # noqa: E402
from inference import InvalidSmilesError  # noqa: E402
from rag.vector_store import IndexCorruptError, RAGError  # noqa: E402


class UiRecoveryTests(unittest.TestCase):
    def tearDown(self) -> None:
        ui_pages.cached_research_index.clear()

    def test_manual_smiles_is_canonicalized_and_label_remains_unverified(self) -> None:
        compound = ui_pages._manual_compound("mygrane", "OCC")
        self.assertEqual(compound["name"], "mygrane")
        self.assertEqual(compound["smiles"], "CCO")
        self.assertFalse(compound["identity_verified"])

    def test_invalid_manual_smiles_is_rejected(self) -> None:
        with self.assertRaises(InvalidSmilesError):
            ui_pages._manual_compound("Unknown", "not_a_smiles")

    def test_prediction_explanation_uses_supplied_outputs_and_handles_missing_values(self) -> None:
        explanation = ui_pages.build_prediction_explanation(
            {
                "HIA": {"prediction": 1, "probability": 0.72},
                "HOB20": {"prediction": 0},
                "HOB50": None,
            }
        )
        self.assertIn("HIA favorable", explanation)
        self.assertIn("HOB20 unfavorable", explanation)
        self.assertIn("incomplete", explanation)
        self.assertNotIn("%", explanation)

    def test_prediction_circle_colors_follow_endpoint_rules(self) -> None:
        self.assertEqual(ui_pages._hia_probability_ring_color(0.70), "#16834A")
        self.assertEqual(ui_pages._hia_probability_ring_color(0.40), "#D99A16")
        self.assertEqual(ui_pages._hia_probability_ring_color(0.39), "#C43D3D")
        self.assertEqual(ui_pages._prediction_ring_color("HOB20", 1, 0.15), "#16834A")
        self.assertEqual(ui_pages._prediction_ring_color("HOB50", 0, 0.95), "#C43D3D")
        self.assertEqual(ui_pages._prediction_ring_color("HIA", 1, None), "#8c99a7")

    def test_prediction_display_adapter_preserves_model_outputs(self) -> None:
        endpoint_outputs = {
            "HIA": {"prediction": 1, "probability": 0.72, "threshold": 0.5},
            "HOB20": {"prediction": 0, "probability": 0.31, "threshold": 0.5},
            "HOB50": {"prediction": 0, "probability": 0.11, "threshold": 0.5},
        }
        warning = "Existing consistency warning"
        display_result = ui_pages._prediction_result_for_display(
            {"predictions": endpoint_outputs, "consistency_warning": warning}
        )
        self.assertIs(display_result["predictions"], endpoint_outputs)
        self.assertEqual(display_result["consistency_warning"], warning)

    def test_valid_saved_index_is_reused_without_rebuild(self) -> None:
        loaded = SimpleNamespace(chunk_count=12)
        with patch.object(ui_pages.ResearchIndex, "load", return_value=loaded) as load, patch.object(
            ui_pages, "build_corpus"
        ) as build:
            index, warning = ui_pages.cached_research_index("valid-index", "model")
        self.assertIs(index, loaded)
        self.assertIsNone(warning)
        load.assert_called_once()
        build.assert_not_called()

    def test_invalid_saved_index_is_rebuilt_and_reloaded(self) -> None:
        loaded = SimpleNamespace(chunk_count=3)
        with patch.object(
            ui_pages.ResearchIndex,
            "load",
            side_effect=[IndexCorruptError("internal details"), loaded],
        ) as load, patch.object(
            ui_pages, "build_corpus", return_value=SimpleNamespace(failed_files=())
        ) as build:
            index, warning = ui_pages.cached_research_index("broken-index", "model")
        self.assertIs(index, loaded)
        self.assertIsNone(warning)
        self.assertEqual(load.call_count, 2)
        build.assert_called_once()

    def test_failed_rebuild_returns_friendly_status_and_is_cached(self) -> None:
        with patch.object(
            ui_pages.ResearchIndex,
            "load",
            side_effect=IndexCorruptError("internal details"),
        ), patch.object(ui_pages, "build_corpus", side_effect=RAGError("internal details")) as build:
            first = ui_pages.cached_research_index("unusable-index", "model")
            second = ui_pages.cached_research_index("unusable-index", "model")
        self.assertEqual(first, second)
        self.assertIsNone(first[0])
        self.assertIn("could not be validated or rebuilt", first[1])
        build.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)

