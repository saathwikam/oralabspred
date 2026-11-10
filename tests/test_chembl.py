from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "p2_chemberta"))

from chembl import (  # noqa: E402
    ChemblAmbiguousError,
    ChemblClient,
    ChemblHTTPError,
    ChemblNotFoundError,
    ChemblServiceError,
    inchikey_from_smiles,
)


ASPIRIN_KEY = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"


class ChemblLookupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = ChemblClient()

    @patch.object(ChemblClient, "_request_json")
    def test_successful_lookup_returns_identifier_and_reported_properties(self, request_json) -> None:
        request_json.return_value = {
            "molecules": [
                {
                    "molecule_chembl_id": "CHEMBL25",
                    "pref_name": "ASPIRIN",
                    "molecule_type": "Small molecule",
                    "molecule_structures": {"standard_inchi_key": ASPIRIN_KEY},
                    "molecule_properties": {
                        "full_molformula": "C9H8O4",
                        "full_mwt": "180.16",
                        "alogp": "1.31",
                        "hba": 4,
                        "hbd": 1,
                        "unreported_value": None,
                    },
                }
            ],
            "page_meta": {"total_count": 1},
        }

        compound = self.client.lookup_by_inchikey(ASPIRIN_KEY.lower())

        self.assertEqual(compound.chembl_id, "CHEMBL25")
        self.assertEqual(compound.preferred_name, "ASPIRIN")
        self.assertEqual(compound.molecular_formula, "C9H8O4")
        self.assertEqual(compound.molecular_weight, 180.16)
        self.assertEqual(compound.reported_properties["alogp"], "1.31")
        self.assertEqual(compound.reported_properties["molecule_type"], "Small molecule")
        self.assertNotIn("full_mwt", compound.reported_properties)
        self.assertTrue(compound.record_url.endswith("/CHEMBL25/"))
        request_json.assert_called_once()

    @patch.object(ChemblClient, "_request_json")
    def test_missing_record_is_not_reported_as_service_failure(self, request_json) -> None:
        request_json.return_value = {"molecules": [], "page_meta": {"total_count": 0}}
        with self.assertRaises(ChemblNotFoundError):
            self.client.lookup_by_inchikey(ASPIRIN_KEY)

    @patch.object(ChemblClient, "_request_json")
    def test_missing_properties_remain_unreported(self, request_json) -> None:
        request_json.return_value = {
            "molecules": [
                {
                    "molecule_chembl_id": "CHEMBL25",
                    "pref_name": None,
                    "molecule_structures": {"standard_inchi_key": ASPIRIN_KEY},
                    "molecule_properties": None,
                }
            ]
        }
        compound = self.client.lookup_by_inchikey(ASPIRIN_KEY)
        self.assertIsNone(compound.preferred_name)
        self.assertIsNone(compound.molecular_formula)
        self.assertIsNone(compound.molecular_weight)
        self.assertEqual(compound.reported_properties, {})

    @patch.object(ChemblClient, "_request_json")
    def test_ambiguous_exact_records_are_not_guessed(self, request_json) -> None:
        request_json.return_value = {
            "molecules": [
                {"molecule_chembl_id": chembl_id, "molecule_structures": {"standard_inchi_key": ASPIRIN_KEY}}
                for chembl_id in ("CHEMBL25", "CHEMBL999")
            ]
        }
        with self.assertRaises(ChemblAmbiguousError):
            self.client.lookup_by_inchikey(ASPIRIN_KEY)

    @patch.object(ChemblClient, "_request_json")
    def test_service_failure_is_propagated(self, request_json) -> None:
        request_json.side_effect = ChemblServiceError("simulated service outage")
        with self.assertRaises(ChemblServiceError):
            self.client.lookup_by_inchikey(ASPIRIN_KEY)

    @patch.object(ChemblClient, "_request_json")
    def test_http_not_found_is_mapped(self, request_json) -> None:
        request_json.side_effect = ChemblHTTPError(404)
        with self.assertRaises(ChemblNotFoundError):
            self.client.lookup_by_inchikey(ASPIRIN_KEY)

    def test_inchikey_is_derived_from_valid_smiles(self) -> None:
        self.assertEqual(inchikey_from_smiles("CC(=O)Oc1ccccc1C(=O)O"), ASPIRIN_KEY)


if __name__ == "__main__":
    unittest.main(verbosity=2)
