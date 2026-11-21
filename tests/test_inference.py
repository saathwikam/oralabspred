from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import inchi

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "p2_chemberta"))

from inference import (  # noqa: E402
    InvalidSmilesError,
    MolecularAbsorptionPredictor,
    consistency_warning,
)


def rows_by_split() -> dict[str, dict[str, str]]:
    data_path = PROJECT_ROOT / "data" / "processed" / "cleaned_dataset.csv"
    split_path = PROJECT_ROOT / "data" / "processed" / "split_manifest.csv"
    with data_path.open(newline="", encoding="utf-8") as stream:
        data = {row["InChIKey"]: row for row in csv.DictReader(stream)}
    with split_path.open(newline="", encoding="utf-8") as stream:
        manifest = {row["InChIKey"]: row["Split"] for row in csv.DictReader(stream)}
    return {
        split: next(row for key, row in data.items() if manifest.get(key) == split)
        for split in ("train", "test")
    }


class TrainedInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.predictor = MolecularAbsorptionPredictor(device="cpu")
        cls.examples = rows_by_split()

    def assert_valid_prediction(self, smiles: str) -> None:
        result = self.predictor.predict(smiles)
        self.assertTrue(result["canonical_smiles"])
        self.assertEqual(set(result["predictions"]), {"HIA", "HOB20", "HOB50"})
        for prediction in result["predictions"].values():
            self.assertGreaterEqual(prediction["probability"], 0.0)
            self.assertLessEqual(prediction["probability"], 1.0)
            self.assertIn(prediction["prediction"], (0, 1))

    def test_training_compound(self) -> None:
        self.assert_valid_prediction(self.examples["train"]["Canonical_SMILES"])

    def test_test_compound(self) -> None:
        self.assert_valid_prediction(self.examples["test"]["Canonical_SMILES"])

    def test_new_valid_compound(self) -> None:
        smiles = "O=C(O)c1ccccc1"  # Benzoic acid, absent from the P1 structure keys.
        molecule = Chem.MolFromSmiles(smiles)
        self.assertIsNotNone(molecule)
        with (PROJECT_ROOT / "data" / "processed" / "cleaned_dataset.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            training_keys = {row["InChIKey"] for row in csv.DictReader(stream)}
        self.assertFalse(
            inchi.MolToInchiKey(molecule) in training_keys,
            "The proposed new-compound test input is already in the P1 dataset.",
        )
        self.assert_valid_prediction(smiles)

    def test_invalid_smiles(self) -> None:
        with self.assertRaises(InvalidSmilesError):
            self.predictor.predict("C1CC")

    def test_consistency_warning_preserves_predictions(self) -> None:
        predictions = {
            "HIA": {"prediction": 1},
            "HOB20": {"prediction": 0},
            "HOB50": {"prediction": 1},
        }
        original = {endpoint: values.copy() for endpoint, values in predictions.items()}
        warning = consistency_warning(predictions)
        self.assertIsNotNone(warning)
        self.assertEqual(predictions, original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
