from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn

V2 = Path(__file__).resolve().parents[1] / "evaluation" / "chemberta_model_validation" / "v2_descriptor_fusion"
sys.path.insert(0, str(V2))
sys.path.insert(0, str(V2.parents[2]))

from descriptors import DESCRIPTOR_NAMES, DescriptorPreprocessor, calculate_descriptors, extract_descriptor_frame
from evaluation.chemberta_model_validation.v2_descriptor_fusion.model import ChemBERTaDescriptorFusionClassifier
from masked_weighted_loss import MaskedWeightedBCELoss


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = nn.Embedding(32, 8)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class DescriptorFusionTests(unittest.TestCase):
    def test_rdkit_descriptor_vector_has_fixed_order_and_finite_values(self):
        values = calculate_descriptors("CC(=O)Oc1ccccc1C(=O)O")
        self.assertEqual(len(values), len(DESCRIPTOR_NAMES))
        self.assertTrue(np.isfinite(values).all())
        self.assertGreater(values[0], 0)

    def test_descriptor_failures_are_audited_and_rows_are_retained(self):
        frame = pd.DataFrame({"InChIKey": ["valid", "invalid"], "Canonical_SMILES": ["CCO", "not smiles"]})
        descriptors, failures = extract_descriptor_frame(frame)
        self.assertEqual(len(descriptors), 2)
        self.assertTrue(descriptors.loc[1, list(DESCRIPTOR_NAMES)].isna().all())
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures.iloc[0].InChIKey, "invalid")

    def test_preprocessor_fits_train_only_and_handles_nan(self):
        train = np.array([[1, 2, 3, 0, 1, 2], [3, 4, 5, 1, 2, 3],
                          [np.nan, 6, 7, 1, 3, 4]], dtype=float)
        prep = DescriptorPreprocessor().fit(train)
        before = prep.parameters()["scaler_mean"]
        transformed = prep.transform(np.array([[10000] * 6], dtype=float))
        self.assertEqual(transformed.shape, (1, 6))
        self.assertTrue(np.isfinite(transformed).all())
        self.assertEqual(prep.parameters()["scaler_mean"], before)
        self.assertEqual(prep.parameters()["fit_split"], "train")

    def test_fusion_forward_shape_and_masked_weighted_loss(self):
        model = ChemBERTaDescriptorFusionClassifier(FakeEncoder(), descriptor_hidden_size=4)
        logits = model(torch.ones((3, 7), dtype=torch.long), torch.ones((3, 7), dtype=torch.long),
                       torch.zeros((3, 6)))
        self.assertEqual(tuple(logits.shape), (3, 3))
        targets = torch.tensor([[1.0, 0.0, float("nan")], [0.0, 1.0, 1.0],
                                [float("nan"), 0.0, 1.0]])
        loss = MaskedWeightedBCELoss([0.2, 0.3, 0.8])(logits, targets)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.descriptor_network[0].weight.grad)

    def test_fusion_rejects_misaligned_descriptors(self):
        model = ChemBERTaDescriptorFusionClassifier(FakeEncoder(), descriptor_hidden_size=4)
        with self.assertRaisesRegex(ValueError, "batch_size"):
            model(torch.ones((2, 7), dtype=torch.long), torch.ones((2, 7), dtype=torch.long),
                  torch.zeros((1, 6)))


if __name__ == "__main__":
    unittest.main()
