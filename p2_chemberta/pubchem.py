from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


class PubChemError(RuntimeError):
    """Base class for compound lookup failures."""


class CompoundNotFoundError(PubChemError):
    """PubChem did not find a compound matching the submitted name."""


class AmbiguousCompoundError(PubChemError):
    """The submitted name resolved to multiple PubChem compound identifiers."""

    def __init__(self, name: str, cids: list[int]) -> None:
        self.name = name
        self.cids = tuple(cids)
        super().__init__(f"'{name}' matched multiple PubChem CIDs: {', '.join(map(str, cids))}.")


class PubChemServiceError(PubChemError):
    """PubChem could not be reached or returned an unusable response."""


class PubChemHTTPError(PubChemServiceError):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"PubChem returned HTTP {status_code}.")


@dataclass(frozen=True)
class PubChemCompound:
    requested_name: str
    cid: int
    canonical_smiles: str
    iupac_name: str | None = None
    inchikey: str | None = None


class PubChemClient:
    """Small PUG REST client for resolving one compound name to a PubChem CID and SMILES."""

    def __init__(self, timeout: float = 12.0) -> None:
        self.timeout = timeout

    def _request_json(self, url: str) -> dict:
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "Oralabs-Predict/1.0 (PubChem compound lookup)",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise PubChemHTTPError(exc.code) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise PubChemServiceError("Could not connect to PubChem.") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PubChemServiceError("PubChem returned an invalid JSON response.") from exc
        if not isinstance(payload, dict):
            raise PubChemServiceError("PubChem returned an unexpected response format.")
        return payload

    def lookup_compound(self, name: str) -> PubChemCompound:
        requested_name = name.strip() if isinstance(name, str) else ""
        if not requested_name:
            raise CompoundNotFoundError("Enter a compound name to search PubChem.")

        encoded_name = quote(requested_name, safe="")
        cid_url = f"{PUBCHEM_BASE_URL}/compound/name/{encoded_name}/cids/JSON"
        try:
            cid_payload = self._request_json(cid_url)
        except PubChemHTTPError as exc:
            if exc.status_code in (400, 404):
                raise CompoundNotFoundError(
                    f"PubChem did not find a compound named '{requested_name}'."
                ) from exc
            raise

        identifier_list = cid_payload.get("IdentifierList")
        if identifier_list is None:
            raise CompoundNotFoundError(f"PubChem did not find a compound named '{requested_name}'.")
        if not isinstance(identifier_list, dict):
            raise PubChemServiceError("PubChem returned an invalid identifier response.")
        raw_cids = identifier_list.get("CID", [])
        if not isinstance(raw_cids, list):
            raise PubChemServiceError("PubChem returned an invalid CID list.")
        try:
            cids = sorted({int(cid) for cid in raw_cids if int(cid) > 0})
        except (TypeError, ValueError) as exc:
            raise PubChemServiceError("PubChem returned an invalid CID value.") from exc
        if not cids:
            raise CompoundNotFoundError(f"PubChem did not find a compound named '{requested_name}'.")
        if len(cids) > 1:
            raise AmbiguousCompoundError(requested_name, cids)

        cid = cids[0]
        property_url = (
            f"{PUBCHEM_BASE_URL}/compound/cid/{cid}/property/"
            "SMILES,ConnectivitySMILES,IUPACName,InChIKey/JSON"
        )
        property_payload = self._request_json(property_url)
        property_table = property_payload.get("PropertyTable")
        if not isinstance(property_table, dict):
            raise PubChemServiceError("PubChem returned an invalid property response.")
        properties = property_table.get("Properties", [])
        if not isinstance(properties, list) or not properties:
            raise PubChemServiceError("PubChem returned no compound properties for the matched CID.")
        record = None
        for item in properties:
            if not isinstance(item, dict):
                continue
            try:
                property_cid = int(item.get("CID", -1))
            except (TypeError, ValueError):
                continue
            if property_cid == cid:
                record = item
                break
        if record is None:
            raise PubChemServiceError("PubChem's property response did not match the requested CID.")

        # Current PUG REST uses SMILES for its stereochemistry-preserving form;
        # ConnectivitySMILES is the canonical connectivity-only fallback.
        smiles = record.get("SMILES") or record.get("ConnectivitySMILES")
        if not isinstance(smiles, str) or not smiles.strip():
            raise PubChemServiceError("PubChem did not return a usable SMILES for the matched CID.")
        iupac_name = record.get("IUPACName")
        inchikey = record.get("InChIKey")
        return PubChemCompound(
            requested_name=requested_name,
            cid=cid,
            canonical_smiles=smiles.strip(),
            iupac_name=iupac_name.strip() if isinstance(iupac_name, str) and iupac_name.strip() else None,
            inchikey=inchikey.strip() if isinstance(inchikey, str) and inchikey.strip() else None,
        )
