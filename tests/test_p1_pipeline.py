"""Focused unit checks for the P1 preprocessing and shared loss interface."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from class_weights import calculate_class_distribution
from data_preprocessing import clean_dataset
from masked_weighted_loss import MaskedWeightedBCELoss
from scaffold_split import create_scaffold_split


class PreprocessingTests(unittest.TestCase):
    def test_invalid_smiles_and_conflicts_are_audited_and_canonicalized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.csv"
            pd.DataFrame([
                {"InChIKey": "untrusted-a", "SMILES": "CCO", "HIA": 1, "HOB20": 1, "HOB50": 0},
                {"InChIKey": "untrusted-b", "SMILES": "OCC", "HIA": 0, "HOB20": 1, "HOB50": 0},
                {"InChIKey": "untrusted-c", "SMILES": "not_a_smiles", "HIA": 1, "HOB20": None, "HOB50": None},
            ]).to_csv(source, index=False)
            stats = clean_dataset(source, root / "processed")
            cleaned = pd.read_csv(root / "processed" / "cleaned_dataset.csv")
            conflicts = pd.read_csv(root / "processed" / "label_conflict_audit.csv")
            invalid = pd.read_csv(root / "processed" / "invalid_records.csv")
            self.assertEqual(stats["valid_smiles"], 2)
            self.assertEqual(stats["invalid_rows"], 1)
            self.assertEqual(len(cleaned), 1)
            self.assertTrue(cleaned.loc[0, "Canonical_SMILES"])
            self.assertTrue(pd.isna(cleaned.loc[0, "HIA"]))
            self.assertEqual(cleaned.loc[0, "HOB20"], 1)
            self.assertEqual(len(conflicts), 1)
            self.assertIn("parse/sanitize", invalid.loc[0, "Exclusion_Reason"])

    def test_ambiguous_label_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.csv"
            pd.DataFrame([{
                "InChIKey": "x", "SMILES": "CC", "HIA": "high", "HOB20": 0, "HOB50": 0,
            }]).to_csv(source, index=False)
            with self.assertRaisesRegex(ValueError, "Ambiguous HIA label"):
                clean_dataset(source, root / "processed")


class SplitAndWeightTests(unittest.TestCase):
    def test_scaffold_groups_do_not_cross_splits_and_seed_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.csv"
            smiles = ["Cc1ccccc1", "Oc1ccccc1", "c1ccncc1", "CCO", "CCCC", "CC(=O)O", "C1CCCCC1"]
            pd.DataFrame([
                {"InChIKey": f"fake-{i}", "SMILES": value, "HIA": i % 2, "HOB20": None, "HOB50": i % 2}
                for i, value in enumerate(smiles)
            ]).to_csv(source, index=False)
            clean_dataset(source, root / "processed")
            manifest, _ = create_scaffold_split(
                root / "processed" / "cleaned_dataset.csv", root / "processed" / "split_manifest.csv", seed=19
            )
            repeat, _ = create_scaffold_split(
                root / "processed" / "cleaned_dataset.csv", root / "processed" / "repeat.csv", seed=19
            )
            self.assertEqual(manifest[["InChIKey", "Split"]].to_dict("records"), repeat[["InChIKey", "Split"]].to_dict("records"))
            self.assertTrue((manifest.groupby("Scaffold_ID")["Split"].nunique() == 1).all())

    def test_weights_are_calculated_from_train_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "cleaned.csv"
            manifest_path = root / "manifest.csv"
            output_path = root / "class_distribution_report.csv"
            pd.DataFrame([
                {"InChIKey": "a", "HIA": 1, "HOB20": 1, "HOB50": 1},
                {"InChIKey": "b", "HIA": 0, "HOB20": 1, "HOB50": None},
                {"InChIKey": "c", "HIA": 0, "HOB20": None, "HOB50": 0},
                {"InChIKey": "d", "HIA": 1, "HOB20": 0, "HOB50": 0},
            ]).to_csv(data_path, index=False)
            pd.DataFrame([
                {"InChIKey": "a", "Split": "train"},
                {"InChIKey": "b", "Split": "train"},
                {"InChIKey": "c", "Split": "train"},
                {"InChIKey": "d", "Split": "validation"},
            ]).to_csv(manifest_path, index=False)
            report, stats = calculate_class_distribution(data_path, manifest_path, output_path)
            self.assertEqual(stats["weights"]["HIA"], 2.0)
            self.assertIsNone(stats["weights"]["HOB20"])
            status = report.query("Endpoint == 'HOB20' and Split == 'train'")["Weight_Status"].iloc[0]
            self.assertEqual(status, "undefined: no negative training labels")
            self.assertEqual(report.query("Endpoint == 'HIA' and Split == 'validation'")["Train_Pos_Weight"].isna().all(), True)


class MaskedWeightedLossTests(unittest.TestCase):
    def test_missing_endpoint_is_skipped_and_gradients_remain_finite(self) -> None:
        criterion = MaskedWeightedBCELoss([1.0, 0.5, 2.0])
        logits = torch.tensor([[0.2, 0.4, -0.3], [0.1, 0.9, 0.2]], requires_grad=True)
        targets = torch.tensor([[1.0, float("nan"), 0.0], [0.0, float("nan"), float("nan")]])
        total, per_task = criterion(logits, targets, return_details=True)
        self.assertTrue(torch.isfinite(total))
        self.assertIsNone(per_task["HOB20"])
        total.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.equal(logits.grad[:, 1], torch.zeros_like(logits.grad[:, 1])))

    def test_all_missing_targets_return_finite_zero_with_gradient_path(self) -> None:
        criterion = MaskedWeightedBCELoss([1.0, 1.0, 1.0])
        logits = torch.randn(2, 3, requires_grad=True)
        targets = torch.full((2, 3), float("nan"))
        total, per_task = criterion(logits, targets, return_details=True)
        self.assertEqual(total.item(), 0.0)
        self.assertTrue(all(value is None for value in per_task.values()))
        total.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()
