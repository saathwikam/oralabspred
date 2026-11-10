from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from rdkit import Chem
from rdkit.Chem import inchi


CHEMBL_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"


class ChemblError(RuntimeError):
    """Base class for ChEMBL retrieval failures."""


class ChemblNotFoundError(ChemblError):
    """No exact ChEMBL molecule record was found for the identifier."""


class ChemblAmbiguousError(ChemblError):
    """More than one ChEMBL molecule matched the exact structure identifier."""

    def __init__(self, inchikey: str, chembl_ids: list[str]) -> None:
        self.inchikey = inchikey
        self.chembl_ids = tuple(chembl_ids)
        super().__init__(f"Multiple ChEMBL molecule records matched {inchikey}.")


class ChemblServiceError(ChemblError):
    """The ChEMBL service failed or returned an unusable response."""


class ChemblHTTPError(ChemblServiceError):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"ChEMBL returned HTTP {status_code}.")


@dataclass(frozen=True)
class ChemblCompound:
    chembl_id: str
    preferred_name: str | None
    molecular_formula: str | None
    molecular_weight: float | None
    reported_properties: dict[str, object]

    @property
    def record_url(self) -> str:
        return f"https://www.ebi.ac.uk/chembl/compound_report_card/{self.chembl_id}/"


def inchikey_from_smiles(smiles: str) -> str:
    """Generate an exact structure key with RDKit for a user-provided SMILES."""
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("A non-empty SMILES is required to identify this compound in ChEMBL.")
    molecule = Chem.MolFromSmiles(smiles.strip())
    if molecule is None:
        raise ValueError("The SMILES could not be validated for ChEMBL identification.")
    try:
        return inchi.MolToInchiKey(molecule)
    except Exception as exc:
        raise ValueError("RDKit could not generate an InChIKey for this SMILES.") from exc


class ChemblClient:
    """Retrieve exact ChEMBL molecule records by standard InChIKey."""

    def __init__(self, timeout: float = 12.0) -> None:
        self.timeout = timeout

    def _request_json(self, url: str) -> dict:
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "Oralabs-Predict/1.0 (ChEMBL molecule lookup)",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise ChemblHTTPError(exc.code) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ChemblServiceError("Could not connect to ChEMBL.") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ChemblServiceError("ChEMBL returned an invalid JSON response.") from exc
        if not isinstance(payload, dict):
            raise ChemblServiceError("ChEMBL returned an unexpected response format.")
        return payload

    def lookup_by_inchikey(self, inchikey: str) -> ChemblCompound:
        key = inchikey.strip().upper() if isinstance(inchikey, str) else ""
        if not key:
            raise ValueError("A non-empty InChIKey is required for ChEMBL lookup.")
        query = urlencode({"molecule_structures__standard_inchi_key": key, "limit": 100})
        url = f"{CHEMBL_BASE_URL}/molecule.json?{query}"
        try:
            payload = self._request_json(url)
        except ChemblHTTPError as exc:
            if exc.status_code == 404:
                raise ChemblNotFoundError(f"No ChEMBL record was found for {key}.") from exc
            raise

        records = payload.get("molecules")
        if records is None:
            raise ChemblServiceError("ChEMBL returned no molecule list.")
        if not isinstance(records, list):
            raise ChemblServiceError("ChEMBL returned an invalid molecule list.")

        exact_records = []
        for record in records:
            if not isinstance(record, dict):
                continue
            structures = record.get("molecule_structures")
            if not isinstance(structures, dict):
                continue
            record_key = structures.get("standard_inchi_key")
            if isinstance(record_key, str) and record_key.upper() == key:
                exact_records.append(record)
        distinct_records = {
            record.get("molecule_chembl_id"): record
            for record in exact_records
            if isinstance(record.get("molecule_chembl_id"), str) and record["molecule_chembl_id"]
        }
        if not distinct_records:
            raise ChemblNotFoundError(f"No exact ChEMBL record was found for {key}.")
        if len(distinct_records) > 1:
            raise ChemblAmbiguousError(key, sorted(distinct_records))

        chembl_id, record = next(iter(distinct_records.items()))
        raw_properties = record.get("molecule_properties")
        properties = raw_properties if isinstance(raw_properties, dict) else {}
        formula_value = properties.get("full_molformula")
        weight_value = properties.get("full_mwt")
        try:
            molecular_weight = float(weight_value) if weight_value is not None else None
        except (TypeError, ValueError):
            molecular_weight = None
        reported_properties = {
            key: value
            for key, value in properties.items()
            if value is not None and value != "" and key not in {"full_molformula", "full_mwt"}
        }
        molecule_type = record.get("molecule_type")
        if molecule_type:
            reported_properties["molecule_type"] = molecule_type

        preferred_name = record.get("pref_name")
        return ChemblCompound(
            chembl_id=chembl_id,
            preferred_name=preferred_name.strip() if isinstance(preferred_name, str) and preferred_name.strip() else None,
            molecular_formula=formula_value.strip() if isinstance(formula_value, str) and formula_value.strip() else None,
            molecular_weight=molecular_weight,
            reported_properties=reported_properties,
        )
