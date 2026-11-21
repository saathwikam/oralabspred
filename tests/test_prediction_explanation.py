from __future__ import annotations

import copy
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "p2_chemberta"))

import prediction_explanation as engine  # noqa: E402


def predictions(hia=1, hob20=1, hob50=1):
    return {"HIA": {"prediction": hia, "probability": 0.82}, "HOB20": {"prediction": hob20}, "HOB50": {"prediction": hob50}}


class DeterministicPredictionExplanationTests(unittest.TestCase):
    def explain(self, outputs=None, descriptors=None, name="Aspirin"):
        return engine.generate_prediction_explanation(
            compound_name=name, smiles="CCO", predictions=outputs if outputs is not None else predictions(),
            descriptors=descriptors if descriptors is not None else {
                "Molecular weight (g/mol)": 610.52,
                "Topological polar surface area (Å²)": 269.43,
                "Hydrogen bond donors": 10,
                "Hydrogen bond acceptors": 16,
            },
        ).as_text()

    def test_low_predictions_include_compound_specific_properties(self):
        text = self.explain(predictions(0, 0, 0))
        self.assertIn("low intestinal absorption", text)
        self.assertIn("unfavorable oral bioavailability", text)
        self.assertIn("610.52 g/mol", text)
        self.assertIn("269.43 Å²", text)
        self.assertIn("not confirmed causes", text)

    def test_all_positive_predictions_summarized(self):
        self.assertIn("favorable intestinal absorption and oral bioavailability", self.explain())

    def test_high_hia_low_hob_explains_distinction(self):
        text = self.explain(predictions(1, 0, 0))
        self.assertIn("does not guarantee favorable oral bioavailability", text)
        self.assertIn("first-pass metabolism", text)

    def test_mixed_labels_are_reported_without_forcing_one_class(self):
        text = self.explain(predictions(0, 1, 0))
        self.assertIn("low intestinal absorption", text)
        self.assertIn("favorable HOB20", text)
        self.assertIn("unfavorable HOB50", text)

    def test_missing_descriptors_are_not_invented(self):
        text = self.explain(descriptors={"XLogP": float("nan")})
        self.assertIn("No usable molecular descriptors", text)
        self.assertNotIn("XLogP 0", text)

    def test_missing_outputs_are_handled(self):
        text = self.explain({"HIA": {"prediction": 1}, "HOB20": {"probability": 0.2}, "HOB50": None}, {})
        self.assertIn("HIA favorable", text)
        self.assertIn("predictions are incomplete", text)

    def test_paragraph_has_two_to_four_sentences_and_no_probabilities_or_subsections(self):
        text = self.explain()
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
        self.assertGreaterEqual(len(sentences), 2)
        self.assertLessEqual(len(sentences), 4)
        self.assertNotIn("%", text)
        self.assertNotIn("**", text)
        for removed in ("Overall interpretation", "Predicted results", "Possible molecular factors", "Understanding the results", "Limitations", "Related local research sources"):
            self.assertNotIn(removed, text)

    def test_distinct_profiles_produce_distinct_text_and_low_logp_is_not_called_bad(self):
        high = self.explain(descriptors={"Molecular weight (g/mol)": 620, "Topological polar surface area (Å²)": 150})
        low_logp = self.explain(predictions(0, 0, 0), {"XLogP": -0.4}, "Example B")
        self.assertNotEqual(high, low_logp)
        self.assertNotIn("poor absorption", low_logp)

    def test_predictions_are_not_mutated(self):
        outputs = predictions(1, 0, 1)
        before = copy.deepcopy(outputs)
        self.explain(outputs)
        self.assertEqual(outputs, before)

    def test_engine_is_local_and_deterministic(self):
        self.assertFalse(hasattr(engine, "answer_with_research"))
        self.assertFalse(hasattr(engine, "client"))
        self.assertEqual(self.explain(), self.explain())

    def test_missing_name_does_not_fabricate_identity(self):
        explanation = engine.generate_prediction_explanation(compound_name="", smiles="CCO", predictions=predictions(), descriptors={})
        self.assertIsNone(explanation.compound_name)
        self.assertIn("For this compound", explanation.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
