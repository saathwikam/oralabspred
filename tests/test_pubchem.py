from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "p2_chemberta"))

from pubchem import (  # noqa: E402
    AmbiguousCompoundError,
    CompoundNotFoundError,
    PubChemClient,
    PubChemHTTPError,
    PubChemServiceError,
)


class PubChemLookupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = PubChemClient()

    @patch.object(PubChemClient, "_request_json")
    def test_successful_lookup_returns_cid_name_and_smiles(self, request_json) -> None:
        request_json.side_effect = [
            {"IdentifierList": {"CID": [2244]}},
            {
                "PropertyTable": {
                    "Properties": [
                        {
                            "CID": 2244,
                            "SMILES": "CC(=O)Oc1ccccc1C(=O)O",
                            "ConnectivitySMILES": "CC(=O)Oc1ccccc1C(O)=O",
                            "IUPACName": "2-acetyloxybenzoic acid",
                            "InChIKey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
                        }
                    ]
                }
            },
        ]

        compound = self.client.lookup_compound("  Aspirin  ")

        self.assertEqual(compound.requested_name, "Aspirin")
        self.assertEqual(compound.cid, 2244)
        self.assertEqual(compound.canonical_smiles, "CC(=O)Oc1ccccc1C(=O)O")
        self.assertEqual(compound.iupac_name, "2-acetyloxybenzoic acid")
        self.assertEqual(compound.inchikey, "BSYNRYMUTXBXSQ-UHFFFAOYSA-N")
        self.assertEqual(request_json.call_count, 2)

    @patch.object(PubChemClient, "_request_json")
    def test_unknown_compound(self, request_json) -> None:
        request_json.return_value = {"IdentifierList": {"CID": []}}
        with self.assertRaises(CompoundNotFoundError):
            self.client.lookup_compound("not a real compound")
        request_json.assert_called_once()

    @patch.object(PubChemClient, "_request_json")
    def test_ambiguous_name_returns_candidate_cids_without_guessing(self, request_json) -> None:
        request_json.return_value = {"IdentifierList": {"CID": [2244, 1983]}}
        with self.assertRaises(AmbiguousCompoundError) as raised:
            self.client.lookup_compound("ambiguous compound")
        self.assertEqual(raised.exception.cids, (1983, 2244))
        request_json.assert_called_once()

    @patch.object(PubChemClient, "_request_json")
    def test_selected_candidate_is_fetched_by_explicit_cid(self, request_json) -> None:
        request_json.return_value = {
            "PropertyTable": {
                "Properties": [{
                    "CID": 2244,
                    "SMILES": "CC(=O)Oc1ccccc1C(=O)O",
                    "IUPACName": "2-acetyloxybenzoic acid",
                    "InChIKey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
                }]
            }
        }
        candidate = self.client.lookup_compound_by_cid("aspirin", 2244)
        self.assertEqual(candidate.cid, 2244)
        self.assertEqual(candidate.requested_name, "aspirin")
        self.assertEqual(candidate.iupac_name, "2-acetyloxybenzoic acid")
        self.assertIn("/compound/cid/2244/property/", request_json.call_args.args[0])

    @patch.object(PubChemClient, "_request_json")
    def test_api_failure_is_reported(self, request_json) -> None:
        request_json.side_effect = PubChemServiceError("simulated network failure")
        with self.assertRaises(PubChemServiceError):
            self.client.lookup_compound("Aspirin")

    @patch.object(PubChemClient, "_request_json")
    def test_pubchem_not_found_http_status_is_mapped(self, request_json) -> None:
        request_json.side_effect = PubChemHTTPError(404)
        with self.assertRaises(CompoundNotFoundError):
            self.client.lookup_compound("unknown")


if __name__ == "__main__":
    unittest.main(verbosity=2)
