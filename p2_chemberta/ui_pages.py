from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:  # Python 3.10
    import toml as tomllib

import streamlit as st
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

from chembl import (
    ChemblAmbiguousError,
    ChemblClient,
    ChemblNotFoundError,
    ChemblServiceError,
    inchikey_from_smiles,
)
from inference import InvalidSmilesError, MolecularAbsorptionPredictor, default_model_dir
from llm_assistant import (
    GeminiAssistantError,
    GeminiConfigurationError,
    get_gemini_api_key,
)
from pubchem import (
    AmbiguousCompoundError,
    CompoundNotFoundError,
    PubChemClient,
    PubChemCompound,
    PubChemServiceError,
)
from prediction_explanation import (
    EXPLANATION_ENGINE_VERSION,
    generate_prediction_explanation,
)
from qa_workflow import DEFAULT_RAG_INDEX_DIR, answer_with_research
from rag.pipeline import build_corpus
from rag.vector_store import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingModelMismatchError,
    IndexCorruptError,
    IndexNotFoundError,
    RAGError,
    ResearchIndex,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DOCUMENTS_DIR = PROJECT_ROOT / "research_documents"
DEFAULT_TOP_K = 5

COMPOUND_LIBRARY: dict[str, tuple[str, ...]] = {
    "Common Drugs": (
        "Aspirin", "Paracetamol", "Ibuprofen", "Metformin", "Naproxen", "Diclofenac",
        "Celecoxib", "Ketoprofen", "Indomethacin", "Sulindac", "Atorvastatin",
        "Omeprazole", "Losartan",
    ),
    "Spices and Plant Compounds": (
        "Curcumin", "Piperine", "Gingerol", "Capsaicin", "Eugenol", "Thymol",
        "Carvacrol", "Cinnamaldehyde", "Allicin", "Menthol",
    ),
    "Flavonoids": (
        "Quercetin", "Kaempferol", "Apigenin", "Luteolin", "Naringenin", "Myricetin",
        "Fisetin", "Hesperetin", "Diosmetin", "Rutin",
    ),
    "Polyphenols": (
        "Resveratrol", "EGCG", "Genistein", "Gallic Acid", "Ferulic Acid",
        "Chlorogenic Acid", "Caffeic Acid", "Syringic Acid", "Ellagic Acid",
        "Pterostilbene",
    ),
    "Vitamins": (
        "Vitamin C", "Niacin", "Riboflavin", "Pyridoxine", "Folic Acid", "Thiamine",
        "Biotin", "Pantothenic Acid", "Vitamin D3", "Vitamin E",
    ),
    "Alkaloids and Methylxanthines": (
        "Caffeine", "Theobromine", "Nicotine", "Theophylline", "Berberine", "Quinine",
        "Papaverine",
    ),
    "Amino Acids and Related Compounds": (
        "Glycine", "L-Glutamine", "Creatine", "Tryptophan", "Leucine", "Isoleucine",
        "Valine", "Tyrosine", "Phenylalanine",
    ),
    "Other Bioactive Compounds": (
        "Melatonin", "Taurine", "Coenzyme Q10", "L-Carnitine", "Spermidine", "Betaine",
        "Inositol", "Urolithin A",
    ),
    "Organic Acids": (
        "Citric Acid", "Malic Acid", "Tartaric Acid", "Succinic Acid", "Fumaric Acid",
    ),
    "Fatty Acids and Lipids": (
        "Oleic Acid", "Linoleic Acid", "Palmitic Acid", "Stearic Acid",
        "Docosahexaenoic Acid (DHA)", "Eicosapentaenoic Acid (EPA)",
    ),
    "Terpenoids": ("Limonene", "Linalool", "Geraniol", "Borneol", "Ursolic Acid"),
    "Carotenoids": (
        "Beta-Carotene", "Lycopene", "Lutein", "Zeaxanthin", "Astaxanthin",
    ),
    "Coumarins and Lignans": (
        "Coumarin", "Esculetin", "Scopoletin", "Sesamin", "Enterolactone",
    ),
    "Organosulfur Compounds": (
        "Sulforaphane", "S-Allyl Cysteine", "Diallyl Disulfide", "Indole-3-Carbinol",
    ),
    "Steroids and Sterols": (
        "Cholesterol", "Beta-Sitosterol", "Stigmasterol", "Ergosterol",
    ),
}

ASSISTANT_QUESTIONS: dict[str, tuple[str, ...]] = {
    "Oral Absorption": (
        "What is human intestinal absorption?",
        "What factors influence intestinal absorption?",
        "How does molecular structure affect oral absorption?",
    ),
    "Molecular Properties": (
        "How do molecular weight and polarity influence absorption?",
        "What is the role of hydrogen bonding in oral absorption?",
        "How does molecular lipophilicity affect absorption?",
    ),
    "Solubility and Permeability": (
        "How do solubility and permeability affect oral absorption?",
        "Why can a compound have poor oral absorption despite good permeability?",
        "What is the relationship between aqueous solubility and bioavailability?",
    ),
    "Bioavailability": (
        "What is oral bioavailability?",
        "What is the difference between absorption and bioavailability?",
        "What is first-pass metabolism?",
        "What factors can reduce oral bioavailability?",
    ),
    "Compound-specific": (
        "Why does curcumin have limited oral bioavailability?",
        "What factors influence the oral absorption of quercetin?",
        "What affects the oral bioavailability of resveratrol?",
        "Why is piperine studied in relation to bioavailability?",
    ),
    "Custom question": (),
}


@st.cache_resource(show_spinner=False)
def load_predictor(model_path: str) -> MolecularAbsorptionPredictor:
    return MolecularAbsorptionPredictor(model_dir=model_path)


@st.cache_data(max_entries=128, show_spinner=False)
def cached_prediction(model_path: str, smiles: str) -> dict[str, Any]:
    return load_predictor(model_path).predict(smiles)


@st.cache_data(ttl=3600, max_entries=128, show_spinner=False)
def cached_compound_lookup(name: str) -> PubChemCompound:
    return PubChemClient().lookup_compound(name)


@st.cache_data(ttl=3600, max_entries=128, show_spinner=False)
def cached_compound_lookup_by_cid(name: str, cid: int) -> PubChemCompound:
    return PubChemClient().lookup_compound_by_cid(name, cid)


@st.cache_data(ttl=3600, max_entries=256, show_spinner=False)
def cached_inchikey(smiles: str) -> str:
    return inchikey_from_smiles(smiles)


@st.cache_data(ttl=3600, max_entries=128, show_spinner=False)
def cached_chembl_lookup(inchikey: str):
    return ChemblClient().lookup_by_inchikey(inchikey)


@st.cache_resource(show_spinner=False)
def cached_research_index(index_dir: str, embedding_model: str) -> tuple[ResearchIndex | None, str | None]:
    """Reuse a valid index or rebuild it once per process when validation fails."""
    try:
        return ResearchIndex.load(index_dir, model_name=embedding_model), None
    except (IndexNotFoundError, IndexCorruptError, EmbeddingModelMismatchError):
        try:
            report = build_corpus(
                RESEARCH_DOCUMENTS_DIR,
                index_dir,
                model_name=embedding_model,
            )
            index = ResearchIndex.load(index_dir, model_name=embedding_model)
            warning = (
                f"The index was rebuilt, but {len(report.failed_files)} research PDF(s) "
                "could not be indexed."
                if report.failed_files
                else None
            )
            return index, warning
        except (RAGError, OSError):
            return None, "Research search is unavailable because the saved index could not be validated or rebuilt."
    except (RAGError, OSError):
        return None, "Research search is unavailable because indexing support is not ready."


@st.cache_data(show_spinner=False, max_entries=1)
def cached_research_catalog() -> list[dict[str, Any]]:
    """Build a searchable catalogue from actual PDFs and the existing chunk metadata."""
    if not RESEARCH_DOCUMENTS_DIR.is_dir():
        return []
    pdf_paths = sorted(
        (path for path in RESEARCH_DOCUMENTS_DIR.rglob("*.pdf") if path.is_file()),
        key=lambda path: path.name.casefold(),
    )
    pdf_names = {path.name for path in pdf_paths}
    records: dict[str, dict[str, Any]] = {
        path.name: {
            "filename": path.name,
            "title": path.stem.replace("_", " "),
            "pages": set(),
            "search_text": path.stem.replace("_", " ").casefold(),
        }
        for path in pdf_paths
    }
    chunks_path = PROJECT_ROOT / "rag_store" / "chunks.json"
    try:
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        chunks = []
    if isinstance(chunks, list):
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            metadata = chunk.get("metadata")
            if not isinstance(metadata, dict):
                continue
            filename = metadata.get("filename")
            if filename not in pdf_names:
                continue
            record = records[filename]
            title = metadata.get("title")
            if isinstance(title, str) and title.strip() and record["title"] == Path(filename).stem.replace("_", " "):
                record["title"] = title.strip()
            page = metadata.get("page_number")
            if isinstance(page, int) and page > 0:
                record["pages"].add(page)
            text = chunk.get("text")
            if isinstance(text, str):
                record["search_text"] += " " + text.casefold()
    for record in records.values():
        record["pages"] = sorted(record["pages"])
    return list(records.values())


def _index_directory() -> Path:
    return Path(os.environ.get("ORALABS_RAG_INDEX_DIR", str(DEFAULT_RAG_INDEX_DIR))).expanduser()


def get_gemini_settings() -> tuple[str | None, str | None]:
    """Resolve one credential/model source, with an app-root secret fallback."""
    try:
        secrets = st.secrets
    except Exception:
        secrets = None
    try:
        api_key = get_gemini_api_key(secrets)
    except Exception:
        api_key = get_gemini_api_key()
    model = None
    if secrets is not None:
        try:
            model = secrets.get("GEMINI_MODEL")
        except Exception:
            model = None
    project_secrets: dict[str, Any] = {}
    if not api_key or not model:
        secrets_path = PROJECT_ROOT / ".streamlit" / "secrets.toml"
        try:
            project_secrets = tomllib.loads(secrets_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    if not api_key:
        api_key = get_gemini_api_key(project_secrets)
    if not model:
        model = project_secrets.get("GEMINI_MODEL")
    return api_key, os.environ.get("GEMINI_MODEL") or model


def _active_compound_context() -> tuple[str, dict[str, Any]]:
    analysis = st.session_state.get("oralabs_analysis")
    if isinstance(analysis, dict):
        info = analysis.get("compound_information")
        name = analysis.get("compound_name") or ""
        if isinstance(info, dict):
            return str(name), info
    selected = st.session_state.get("selected_compound")
    if isinstance(selected, dict):
        name = str(selected.get("name") or "")
        return name, {
            "PubChem": {
                "CID": selected.get("cid"),
                "IUPAC name": selected.get("iupac_name"),
                "canonical SMILES": selected.get("smiles"),
            }
        }
    return "", {}


def _group_sources(passages) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int | None], dict[str, Any]] = {}
    for passage in passages:
        key = (str(passage.filename), passage.page_number)
        record = grouped.setdefault(
            key,
            {
                "filename": str(passage.filename),
                "page_number": passage.page_number,
                "title": passage.title,
                "source_reference": passage.source_reference,
                "passages": [],
            },
        )
        if passage.text and passage.text not in record["passages"]:
            record["passages"].append(passage.text)
    return list(grouped.values())


def _verified_source(source_records: list[dict[str, Any]], filename: str, page: int) -> bool:
    return any(
        Path(record["filename"]).name == filename and record["page_number"] == page
        for record in source_records
    )


def _set_pdf_page(filename: str, page_number: int | None) -> None:
    safe_name = Path(filename).name
    target = RESEARCH_DOCUMENTS_DIR / safe_name
    if target.is_file() and target.suffix.lower() == ".pdf":
        st.session_state["research_pdf_selection"] = {
            "filename": safe_name,
            "page_number": page_number,
        }


def _render_pdf_selection(key_prefix: str) -> None:
    selection = st.session_state.get("research_pdf_selection")
    if not isinstance(selection, dict):
        return
    filename = str(selection.get("filename") or "")
    path = RESEARCH_DOCUMENTS_DIR / Path(filename).name
    if not path.is_file() or path.suffix.lower() != ".pdf":
        st.info("The selected research PDF is no longer available in the project collection.")
        return
    page_number = selection.get("page_number")
    st.subheader("Research PDF")
    page_bytes: bytes | None = None
    if isinstance(page_number, int):
        st.caption(f"{filename} · cited page {page_number}")
        try:
            import pymupdf

            with pymupdf.open(path) as original:
                if page_number < 1 or page_number > original.page_count:
                    st.warning("The cited page is outside this PDF's page range.")
                else:
                    one_page = pymupdf.open()
                    one_page.insert_pdf(
                        original, from_page=page_number - 1, to_page=page_number - 1
                    )
                    page_bytes = one_page.tobytes()
                    one_page.close()
        except (ImportError, OSError, RuntimeError, ValueError):
            page_bytes = None
        if page_bytes:
            st.download_button(
                f"Download cited page {page_number}",
                data=page_bytes,
                file_name=f"{path.stem}_page_{page_number}.pdf",
                mime="application/pdf",
                key=f"{key_prefix}_cited_page_download",
            )
        else:
            st.info("The cited page can be opened from the original PDF after download.")
    else:
        st.info("Download the selected PDF to open it in your device's PDF viewer.")
    st.download_button(
        "Download original PDF",
        data=path.read_bytes(),
        file_name=filename,
        mime="application/pdf",
        key=f"{key_prefix}_original_download",
    )


def render_dashboard(page_links: list[tuple[Any, str, str]]) -> None:
    st.title("Welcome to OraLabs Predict")
    st.caption("Explore molecular absorption predictions and research insights.")
    st.markdown("### Explore MolInsight")
    st.caption("Each section has a separate purpose; predictions run only when you submit them.")
    columns = st.columns(2, gap="medium")
    descriptions = {
        "Prediction": "Validate a compound structure and run the bundled ChemBERTa V1 model.",
        "Compounds": "Search a curated compound list and verify structures through PubChem.",
        "AI Assistant": "Ask scientific questions using the local research collection and Gemini.",
        "Research": "Search and read the original PDFs included with this project.",
    }
    for idx, (page, name, icon) in enumerate(page_links):
        with columns[idx % 2].container(border=True):
            st.markdown(f"#### {icon} {name}")
            st.write(descriptions[name])
            st.page_link(page, label=f"Open {name}", width="content")


def _serialize_compound(record: PubChemCompound, canonical_smiles: str) -> dict[str, Any]:
    return {
        "name": record.requested_name,
        "cid": record.cid,
        "smiles": canonical_smiles,
        "iupac_name": record.iupac_name,
        "inchikey": record.inchikey,
        "record_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{record.cid}",
    }


def _reset_library_selection() -> None:
    """Drop category-dependent selection and any structure from the prior category."""
    for key in (
        "library_compound_choice",
        "library_candidate_name",
        "library_verified_compound",
        "library_validation_attempted_for",
        "library_validation_error",
        "library_compound_name_input",
        "selected_compound",
        "pending_prediction_compound",
        "library_ambiguous_cids",
        "library_ambiguous_name",
    ):
        st.session_state.pop(key, None)


def _invalidate_edited_library_name() -> None:
    """Editing the displayed name invalidates the structure verified for its old name."""
    st.session_state.pop("library_verified_compound", None)
    st.session_state.pop("library_validation_error", None)
    st.session_state.pop("selected_compound", None)
    st.session_state.pop("pending_prediction_compound", None)
    st.session_state.pop("library_ambiguous_cids", None)
    st.session_state.pop("library_ambiguous_name", None)


def _resolve_library_compound(name: str) -> dict[str, Any]:
    """Resolve the name through PubChem and validate its returned SMILES with RDKit."""
    record = cached_compound_lookup(name.strip())
    canonical_smiles = MolecularAbsorptionPredictor.canonicalize_smiles(
        record.canonical_smiles
    )
    if not record.cid or not canonical_smiles:
        raise InvalidSmilesError("PubChem did not provide a complete compound structure record.")
    return _serialize_compound(record, canonical_smiles)


def _manual_compound(name: str, smiles: str) -> dict[str, Any]:
    canonical_smiles = MolecularAbsorptionPredictor.canonicalize_smiles(smiles)
    return {
        "name": name.strip() or "User-provided compound",
        "smiles": canonical_smiles,
        "identity_verified": False,
    }


def render_compounds_page(prediction_page: Any) -> None:
    st.title("Compounds")
    st.caption("Browse compounds and select one for absorption prediction.")
    categories = list(COMPOUND_LIBRARY)
    category = st.selectbox(
        "Choose category",
        categories,
        index=None,
        placeholder="Choose a category",
        key="library_category",
        on_change=_reset_library_selection,
    )
    names = COMPOUND_LIBRARY.get(category, ())
    selected_name = st.selectbox(
        "Choose compound",
        names,
        index=None,
        placeholder="Choose a compound",
        key="library_compound_choice",
        disabled=category is None,
    )

    # Resolve a newly selected library entry once. Cached lookups prevent repeat
    # PubChem requests on ordinary Streamlit reruns.
    if selected_name and selected_name != st.session_state.get("library_candidate_name"):
        st.session_state["library_candidate_name"] = selected_name
        st.session_state["library_compound_name_input"] = selected_name
        st.session_state.pop("library_verified_compound", None)
        st.session_state.pop("library_validation_error", None)
        st.session_state.pop("selected_compound", None)
        st.session_state.pop("pending_prediction_compound", None)
        st.session_state.pop("library_validation_attempted_for", None)
        st.session_state.pop("library_ambiguous_cids", None)
        st.session_state.pop("library_ambiguous_name", None)

    if (
        selected_name
        and selected_name != st.session_state.get("library_validation_attempted_for")
    ):
        try:
            with st.spinner("Verifying the selected name with PubChem and RDKit..."):
                verified = _resolve_library_compound(selected_name)
            st.session_state["library_verified_compound"] = verified
            st.session_state["library_validation_attempted_for"] = selected_name
            st.session_state.pop("library_validation_error", None)
            st.success(
                f"Verified {verified['name']} · PubChem CID {verified['cid']} · "
                "SMILES validated by RDKit."
            )
        except AmbiguousCompoundError as exc:
            st.session_state["library_validation_attempted_for"] = selected_name
            st.session_state["library_ambiguous_cids"] = exc.cids
            st.session_state["library_ambiguous_name"] = selected_name
            st.session_state["library_validation_error"] = (
                f"PubChem returned multiple possible identities (CIDs: "
                f"{', '.join(map(str, exc.cids[:5]))}). Choose or enter a more specific name."
            )
        except CompoundNotFoundError:
            st.session_state["library_validation_attempted_for"] = selected_name
            st.session_state["library_validation_error"] = (
                "PubChem did not find this compound name. No structure is selected."
            )
        except (PubChemServiceError, InvalidSmilesError) as exc:
            st.session_state["library_validation_attempted_for"] = selected_name
            st.session_state["library_validation_error"] = (
                f"The compound could not be verified: {exc}"
            )

    compound_name = st.text_input(
        "Enter compound name",
        key="library_compound_name_input",
        on_change=_invalidate_edited_library_name,
    )
    manual_smiles = st.text_input(
        "Manual SMILES (for names PubChem cannot resolve)",
        placeholder="Paste a SMILES string to validate with RDKit",
        key="library_manual_smiles",
    )
    verified = st.session_state.get("library_verified_compound")
    if st.session_state.get("library_validation_error"):
        st.warning(st.session_state["library_validation_error"])
    candidate_cids = st.session_state.get("library_ambiguous_cids", ())
    candidate_name = st.session_state.get("library_ambiguous_name", "")
    if candidate_cids and compound_name.strip() == candidate_name:
        candidate_cid = st.selectbox(
            "Choose a possible PubChem match",
            candidate_cids,
            format_func=lambda cid: f"PubChem CID {cid}",
            key="library_ambiguous_cid_choice",
        )
        try:
            candidate = cached_compound_lookup_by_cid(candidate_name, candidate_cid)
            st.caption(
                f"Candidate: {candidate.iupac_name or candidate_name} · "
                f"CID {candidate.cid} · verify this is the intended compound."
            )
            if st.button("Use selected PubChem match", key="library_use_ambiguous_match"):
                verified = _serialize_compound(
                    candidate,
                    MolecularAbsorptionPredictor.canonicalize_smiles(candidate.canonical_smiles),
                )
                st.session_state["library_verified_compound"] = verified
                st.session_state["selected_compound"] = verified
                st.session_state["pending_prediction_compound"] = verified
                st.session_state["pending_prediction_token"] = (
                    int(st.session_state.get("pending_prediction_token", 0)) + 1
                )
                st.switch_page(prediction_page)
        except (PubChemServiceError, InvalidSmilesError) as exc:
            st.warning(f"The selected PubChem candidate could not be verified: {exc}")
    if (
        selected_name
        and compound_name.strip().casefold() != selected_name.strip().casefold()
        and not verified
    ):
        st.caption(
            "The edited name has no verified structure yet. The action below will perform a fresh "
            "PubChem lookup and RDKit validation."
        )

    if st.button(
        "Use this compound for prediction",
        type="primary",
        key="library_use_for_prediction",
        disabled=not compound_name.strip(),
    ):
        try:
            verified = st.session_state.get("library_verified_compound")
            if not verified or compound_name.strip().casefold() != str(
                verified.get("name", "")
            ).strip().casefold():
                with st.spinner("Verifying the entered name and structure..."):
                    verified = _resolve_library_compound(compound_name)
                st.session_state["library_verified_compound"] = verified
            st.session_state["selected_compound"] = verified
            st.session_state["pending_prediction_compound"] = verified
            st.session_state["pending_prediction_token"] = (
                int(st.session_state.get("pending_prediction_token", 0)) + 1
            )
            st.switch_page(prediction_page)
        except AmbiguousCompoundError as exc:
            st.session_state["library_ambiguous_cids"] = exc.cids
            st.session_state["library_ambiguous_name"] = compound_name.strip()
            st.warning(
                "PubChem returned multiple possible identities (CIDs: "
                f"{', '.join(map(str, exc.cids[:5]))}). Choose a candidate above."
            )
        except CompoundNotFoundError:
            st.warning("PubChem did not find this name. No structure was sent to Prediction.")
        except (PubChemServiceError, InvalidSmilesError) as exc:
            st.warning(f"The compound could not be verified: {exc}")

    if st.button(
        "Use manual SMILES for prediction",
        key="library_use_manual_smiles",
        disabled=not manual_smiles.strip(),
    ):
        try:
            manual_compound = _manual_compound(compound_name, manual_smiles)
            st.session_state["selected_compound"] = manual_compound
            st.session_state["pending_prediction_compound"] = manual_compound
            st.session_state["pending_prediction_token"] = (
                int(st.session_state.get("pending_prediction_token", 0)) + 1
            )
            st.switch_page(prediction_page)
        except InvalidSmilesError as exc:
            st.warning(f"The manual SMILES is invalid: {exc}")


def _hia_probability_ring_color(probability: float) -> str:
    if probability >= 0.70:
        return "#16834A"
    if probability >= 0.40:
        return "#D99A16"
    return "#C43D3D"


def _prediction_ring_color(endpoint: str, prediction: int, probability: float | None) -> str:
    if endpoint == "HIA":
        return _hia_probability_ring_color(probability) if probability is not None else "#8c99a7"
    return "#16834A" if prediction == 1 else "#C43D3D"


def _prediction_result_for_display(model_output: dict[str, Any]) -> dict[str, Any]:
    """Keep inference metadata separate from the endpoint values consumed by the UI."""
    return {
        "predictions": model_output["predictions"],
        "consistency_warning": model_output.get("consistency_warning"),
    }


def _hob_count_ring_color(positive_count: int, inconsistent: bool) -> str:
    if inconsistent:
        return "#D97706"
    return {2: "#16834A", 1: "#D99A16", 0: "#C43D3D"}[positive_count]


def _result_circle(
    title: str,
    display: str,
    detail: str,
    value: float | None = None,
    ring_color: str | None = None,
) -> None:
    if value is None:
        fill = "#eaf0f5"
        accent = "#4e789b"
    else:
        percent = max(0.0, min(1.0, value)) * 100.0
        accent = ring_color or "#477da5"
        fill = f"conic-gradient({accent} {percent:.2f}%, #e4ebf1 {percent:.2f}%)"
    st.markdown(
        f"""<div class="mi-result-card">
          <div class="mi-circle" style="background:{fill};--mi-accent:{accent}">
            <div class="mi-circle-inner"><strong>{display}</strong><span>{detail}</span></div>
          </div>
          <div class="mi-circle-label">{title}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _compound_information(analysis: dict[str, Any]) -> None:
    st.subheader("Compound Information")
    pubchem = analysis.get("compound_record")
    chembl = analysis.get("chembl_record")
    left, right = st.columns(2, gap="medium")
    with left.container(border=True):
        st.markdown("#### PubChem")
        if pubchem:
            st.write(f"**CID:** {pubchem.get('cid', 'Not reported')}")
            if pubchem.get("record_url"):
                st.markdown(f"[Open PubChem record]({pubchem['record_url']})")
            st.write(f"**IUPAC name:** {pubchem.get('iupac_name') or 'Not reported'}")
            st.write(f"**InChIKey:** {pubchem.get('inchikey') or 'Not reported'}")
        else:
            st.caption("PubChem name lookup was not requested for this SMILES input.")
    with right.container(border=True):
        st.markdown("#### ChEMBL")
        if analysis.get("chembl_status") == "found" and chembl is not None:
            st.markdown(f"[{chembl.chembl_id}]({chembl.record_url})")
            st.write(f"**Preferred name:** {chembl.preferred_name or 'Not reported'}")
            st.write(f"**Molecular formula:** {chembl.molecular_formula or 'Not reported'}")
            weight = (
                f"{chembl.molecular_weight:g} g/mol"
                if chembl.molecular_weight is not None
                else "Not reported"
            )
            st.write(f"**Molecular weight:** {weight}")
            if chembl.reported_properties:
                st.markdown("**Other properties reported by ChEMBL**")
                for key, value in sorted(chembl.reported_properties.items()):
                    st.write(f"{key.replace('_', ' ').capitalize()}: {value}")
        elif analysis.get("chembl_status") == "not_found":
            st.caption("No exact ChEMBL record was found for this structure.")
        else:
            st.caption("ChEMBL information is temporarily unavailable or was not reported.")


def render_prediction_page() -> None:
    pending = st.session_state.get("pending_prediction_compound")
    token = st.session_state.get("pending_prediction_token")
    if pending and token != st.session_state.get("applied_prediction_token"):
        st.session_state["prediction_input_mode"] = "SMILES"
        st.session_state["prediction_smiles"] = pending["smiles"]
        st.session_state["prediction_name"] = pending["name"]
        st.session_state["applied_prediction_token"] = token

    st.title("Prediction")
    st.caption("Predict HIA, HOB20 and HOB50 using a compound name or SMILES.")
    mode = st.radio(
        "Input type",
        ("Compound name", "SMILES"),
        horizontal=True,
        key="prediction_input_mode",
    )
    if mode == "Compound name":
        raw_value = st.text_input(
            "Enter compound name",
            placeholder="Enter a compound name",
            key="prediction_name",
        )
    else:
        raw_value = st.text_input(
            "Enter SMILES string",
            placeholder="Enter a compound SMILES",
            key="prediction_smiles",
        )
        selected = st.session_state.get("selected_compound")
        if (
            isinstance(selected, dict)
            and selected.get("cid")
            and selected.get("smiles") == raw_value.strip()
        ):
            st.caption(
                f"Verified compound: {selected.get('name')} · PubChem CID {selected.get('cid')}"
            )
    if st.button("Predict", type="primary", key="prediction_submit"):
        st.session_state.pop("oralabs_analysis", None)
        if not raw_value.strip():
            st.warning("Enter a compound name or SMILES string first.")
        else:
            try:
                compound_record: PubChemCompound | None = None
                selected = st.session_state.get("selected_compound")
                with st.spinner("Validating the structure and running ChemBERTa V1..."):
                    if mode == "Compound name":
                        compound_record = cached_compound_lookup(raw_value.strip())
                        smiles = compound_record.canonical_smiles
                    else:
                        smiles = raw_value.strip()
                        if (
                            isinstance(selected, dict)
                            and selected.get("cid")
                            and selected.get("smiles") == smiles
                        ):
                            compound_record = PubChemCompound(
                                requested_name=selected["name"],
                                cid=selected["cid"],
                                canonical_smiles=selected["smiles"],
                                iupac_name=selected.get("iupac_name"),
                                inchikey=selected.get("inchikey"),
                            )
                    canonical = MolecularAbsorptionPredictor.canonicalize_smiles(smiles)
                    result = cached_prediction(str(default_model_dir()), canonical)
                    inchikey = (
                        compound_record.inchikey
                        if compound_record and compound_record.inchikey
                        else cached_inchikey(canonical)
                    )
                    try:
                        chembl_record = cached_chembl_lookup(inchikey)
                        chembl_status = "found"
                    except ChemblNotFoundError:
                        chembl_record, chembl_status = None, "not_found"
                    except (ChemblAmbiguousError, ChemblServiceError, ValueError):
                        chembl_record, chembl_status = None, "unavailable"
                compound_info: dict[str, Any] = {}
                if compound_record:
                    compound_info["PubChem"] = {
                        "CID": compound_record.cid,
                        "IUPAC name": compound_record.iupac_name,
                        "canonical SMILES": canonical,
                        "InChIKey": compound_record.inchikey,
                        "record URL": f"https://pubchem.ncbi.nlm.nih.gov/compound/{compound_record.cid}",
                    }
                else:
                    compound_info["PubChem lookup"] = "Not queried; a SMILES string was supplied directly."
                    selected = st.session_state.get("selected_compound")
                    if (
                        isinstance(selected, dict)
                        and selected.get("smiles") == canonical
                        and not selected.get("identity_verified", True)
                    ):
                        compound_info["Compound label"] = (
                            f"{selected.get('name')}; user-provided and not verified by PubChem."
                        )
                if chembl_record:
                    compound_info["ChEMBL"] = {
                        "molecule ID": chembl_record.chembl_id,
                        "preferred name": chembl_record.preferred_name,
                        "molecular formula": chembl_record.molecular_formula,
                        "molecular weight": chembl_record.molecular_weight,
                        "reported properties": chembl_record.reported_properties,
                        "record URL": chembl_record.record_url,
                    }
                selected_compound = st.session_state.get("selected_compound")
                unverified_label = (
                    str(selected_compound.get("name", ""))
                    if isinstance(selected_compound, dict)
                    and selected_compound.get("smiles") == canonical
                    and not selected_compound.get("identity_verified", True)
                    else ""
                )
                st.session_state["oralabs_analysis"] = {
                    "input_mode": mode,
                    "input_value": raw_value.strip(),
                    "compound_name": (
                        compound_record.requested_name
                        if compound_record
                        else unverified_label
                    ),
                    "identity_verified": compound_record is not None or not unverified_label,
                    "compound_record": _serialize_compound(compound_record, canonical)
                    if compound_record else None,
                    "canonical_smiles": canonical,
                    "compound_information": compound_info,
                    "chembl_record": chembl_record,
                    "chembl_status": chembl_status,
                    "result": result,
                }
                st.session_state["selected_compound"] = (
                    _serialize_compound(compound_record, canonical) if compound_record else None
                )
            except AmbiguousCompoundError as exc:
                st.error(
                    f"PubChem found multiple records (CIDs: {', '.join(map(str, exc.cids[:5]))}). "
                    "Enter a more specific compound name."
                )
            except CompoundNotFoundError:
                st.error("PubChem did not find that compound name.")
            except (PubChemServiceError, InvalidSmilesError) as exc:
                st.error(str(exc))
            except FileNotFoundError:
                st.error("The bundled ChemBERTa V1 model files are unavailable.")

    analysis = st.session_state.get("oralabs_analysis")
    if not isinstance(analysis, dict):
        return
    if analysis.get("input_mode") != mode or analysis.get("input_value") != raw_value.strip():
        return

    st.success(f"Structure validated · canonical SMILES: `{analysis['canonical_smiles']}`")
    if analysis.get("compound_name"):
        label_status = " (user-provided label; identity not verified)" if not analysis.get("identity_verified") else ""
        st.caption(f"Compound: {analysis['compound_name']}{label_status}")
    st.subheader("ChemBERTa prediction results")
    preds = analysis["result"]["predictions"]
    hia = preds["HIA"]
    hob20, hob50 = preds["HOB20"], preds["HOB50"]
    inconsistent = hob50["prediction"] == 1 and hob20["prediction"] == 0
    positive_hob_count = int(hob20["prediction"] == 1) + int(hob50["prediction"] == 1)
    hia_probability = float(hia["probability"])
    hia_ring_color = _hia_probability_ring_color(hia_probability)
    hob_ring_color = _hob_count_ring_color(positive_hob_count, inconsistent)
    hia_class = "Positive" if hia["prediction"] else "Negative"
    first, second = st.columns(2, gap="large")
    with first:
        _result_circle(
            "Human Intestinal Absorption (HIA)",
            f"{hia['probability'] * 100:.1f}%",
            hia_class,
            hia_probability,
            hia_ring_color,
        )
    with second:
        _result_circle(
            "Positive HOB classifications",
            f"{positive_hob_count}/2",
            "Derived endpoint count",
            positive_hob_count / 2,
            hob_ring_color,
        )
        st.caption("Derived from HOB20 and HOB50 classifications; not a separate model prediction.")

    hob_left, hob_right = st.columns(2, gap="medium")
    for column, endpoint in ((hob_left, "HOB20"), (hob_right, "HOB50")):
        value = preds[endpoint]
        classification = "Positive" if value["prediction"] else "Negative"
        with column.container(border=True):
            st.markdown(f"#### {endpoint}")
            st.metric("Predicted probability", f"{value['probability'] * 100:.1f}%")
            st.write(f"**Classification:** {classification}")
            st.caption(f"Model decision cutoff: {value['threshold']:.2f} probability")
    if inconsistent:
        st.warning(
            "HOB20 is negative while HOB50 is positive. The circle counts classifications only; "
            "both model outputs are shown unchanged."
        )
    _compound_information(analysis)


def _research_category(record: dict[str, Any]) -> str:
    text = (record["title"] + " " + record["filename"]).casefold()
    if any(word in text for word in ("curcumin", "turmeric")):
        return "Curcumin and turmeric"
    if any(word in text for word in ("quercetin", "flavonoid", "kaempferol", "naringenin")):
        return "Flavonoids and polyphenols"
    if any(word in text for word in ("bioequivalence", "pharmacokinetic", "pharmacokinetics")):
        return "Drug pharmacokinetics"
    if any(word in text for word in ("formulation", "bioavailability", "absorption")):
        return "Formulation and bioavailability"
    return "Other research"


def render_research_page() -> None:
    st.markdown('<div class="mi-brand">MolInsight <span>RESEARCH</span></div>', unsafe_allow_html=True)
    st.title("Research Library")
    st.caption("Explore the research papers included in this project.")
    catalog = cached_research_catalog()
    if not catalog:
        st.warning("No research PDFs were found in the project collection.")
        return
    with st.spinner("Checking the research index..."):
        index, index_warning = cached_research_index(
            str(_index_directory()), DEFAULT_EMBEDDING_MODEL
        )
    chunk_count: int | None = index.chunk_count if index is not None else None
    metric_col, chunk_col = st.columns(2)
    metric_col.metric("Research PDFs", len(catalog))
    if chunk_count is not None:
        chunk_col.metric("Indexed chunks", chunk_count)
    else:
        chunk_col.caption("Indexed chunk count is unavailable.")
        st.warning(index_warning or "Research search is currently unavailable.")
    if index_warning and chunk_count is not None:
        st.warning(index_warning)

    search = st.text_input(
        "Search titles or indexed research text",
        placeholder="Try curcumin, formulation, absorption...",
        key="research_search",
    )
    categories = sorted({_research_category(record) for record in catalog})
    category = st.selectbox("Filter by document topic", ["All topics", *categories], key="research_category")
    shown = [
        row for row in catalog
        if (category == "All topics" or _research_category(row) == category)
        and (
            not search.strip()
            or search.casefold().strip() in row["search_text"]
            or search.casefold().strip() in row["title"].casefold()
            or search.casefold().strip() in row["filename"].casefold()
        )
    ]
    st.caption(f"Showing {len(shown)} of {len(catalog)} documents.")
    if not shown:
        st.info("No documents match this search and topic filter.")
        return
    labels = {f"{row['title']} · {row['filename']}": row for row in shown}
    selected_label = st.selectbox("Choose a research document", list(labels), key="research_document")
    selected = labels[selected_label]
    st.markdown(f"### {selected['title']}")
    st.caption(selected["filename"])
    pages = selected["pages"]
    if pages:
        st.caption(
            f"Pages with indexed text ({len(pages)}): "
            + ", ".join(map(str, pages))
        )
    path = RESEARCH_DOCUMENTS_DIR / Path(selected["filename"]).name
    if not path.is_file():
        st.warning("The selected PDF is missing from the project folder.")
        return
    data = path.read_bytes()
    st.info("Download the original PDF to open it in your device's PDF viewer.")
    st.download_button(
        "Download selected PDF",
        data=data,
        file_name=path.name,
        mime="application/pdf",
        key="research_pdf_download",
    )


def render_ai_assistant() -> None:
    st.title("AI Assistant")
    st.caption("Ask scientific questions about oral absorption and bioavailability.")
    category = st.selectbox("Question category", list(ASSISTANT_QUESTIONS), key="assistant_category")
    if category == "Custom question":
        question = st.text_area(
            "Your question",
            placeholder="Ask any scientific question about oral absorption or bioavailability...",
            key="assistant_custom_question",
        )
    else:
        question = st.selectbox(
            "Choose a question",
            ASSISTANT_QUESTIONS[category],
            key=f"assistant_question_{category}",
        )
    submitted = st.button("Ask Assistant", type="primary", key="assistant_submit")
    if submitted:
        st.session_state.pop("assistant_answer", None)
        st.session_state.pop("assistant_error", None)
        st.session_state.pop("assistant_error_type", None)
        st.session_state.pop("assistant_retrieval_error", None)
        st.session_state.pop("assistant_sources", None)
        if not question.strip():
            st.warning("Enter or select a question first.")
        else:
            api_key, model_name = get_gemini_settings()
            compound_name, compound_info = _active_compound_context()
            try:
                index_dir = _index_directory()
                with st.spinner("Searching research sources and generating an answer..."):
                    index, index_warning = cached_research_index(
                        str(index_dir), DEFAULT_EMBEDDING_MODEL
                    )
                    index_load_error = None if index is not None else "ResearchIndexUnavailable"
                    result = answer_with_research(
                        compound_name=compound_name,
                        compound_information=compound_info,
                        question=question,
                        # No ChemBERTa predictions are passed to scientific Q&A.
                        api_key=api_key if api_key is not None else "",
                        gemini_model=model_name,
                        index_dir=index_dir,
                        top_k=DEFAULT_TOP_K,
                        research_index=index,
                        research_index_error=index_load_error,
                    )
                st.session_state["assistant_answer"] = result.answer
                st.session_state["assistant_error"] = result.assistant_error
                st.session_state["assistant_error_type"] = result.assistant_error_type
                st.session_state["assistant_retrieval_error"] = result.retrieval_error
                st.session_state["assistant_sources"] = _group_sources(result.research_passages)
            except ValueError as exc:
                st.session_state["assistant_error"] = str(exc)

    api_key, _ = get_gemini_settings()
    error = st.session_state.get("assistant_error")
    error_type = st.session_state.get("assistant_error_type")
    if (
        api_key
        and error_type == "GeminiConfigurationError"
        and error == "Gemini API key is not configured."
    ):
        # A previous run may have stored a missing-key message before the
        # project's secrets were loaded. Do not keep showing it once this
        # process resolves the configured credential successfully.
        st.session_state.pop("assistant_error", None)
        st.session_state.pop("assistant_error_type", None)
        error = None
    if error:
        if error_type == "GeminiConfigurationError" and not api_key:
            st.warning(
                "Gemini Q&A is not configured. Add GEMINI_API_KEY to the environment or "
                "Streamlit secrets."
            )
        else:
            st.warning(error)
    answer = st.session_state.get("assistant_answer")
    sources = st.session_state.get("assistant_sources", [])
    if answer:
        st.subheader("Answer")
        st.markdown(answer)
    if st.session_state.get("assistant_retrieval_error"):
        st.info("Research retrieval was unavailable. This response has no retrieved PDF evidence.")

    if sources:
        st.subheader("Research Evidence")
        for i, source in enumerate(sources):
            label = f"{source['filename']}, p. {source['page_number']}" if source.get("page_number") else source["filename"]
            with st.container(border=True):
                st.markdown(f"**{source.get('title') or source['filename']}** · {label}")
                for text in source["passages"]:
                    st.write(text)
        st.subheader("Research Sources")
        for i, source in enumerate(sources):
            filename = Path(source["filename"]).name
            page = source.get("page_number")
            label = f"{filename}, p. {page}" if page else filename
            cols = st.columns([4, 1])
            cols[0].write(source.get("title") or filename)
            if page:
                cols[0].caption(label)
            if cols[1].button("Open PDF page" if page else "Open PDF", key=f"assistant_source_{i}"):
                _set_pdf_page(filename, page)
                st.rerun()

        # Turn only citations that match retrieved filename/page pairs into controls.
        citation_labels = re.findall(r"\[([^\]]+\.pdf,\s*p\.\s*\d+)\]", answer or "", flags=re.I)
        unique_citations = list(dict.fromkeys(citation_labels))
        valid_citations: list[tuple[str, str, int]] = []
        for citation in unique_citations:
            match = re.fullmatch(r"(.+?\.pdf),\s*p\.\s*(\d+)", citation, flags=re.I)
            if match:
                filename, page_text = match.groups()
                page = int(page_text)
                if _verified_source(sources, filename, page):
                    valid_citations.append((citation, filename, page))
        for idx, (citation, filename, page) in enumerate(valid_citations):
            if st.button(f"View cited evidence: {citation}", key=f"assistant_citation_{idx}"):
                _set_pdf_page(filename, page)
                st.rerun()
    elif answer and not st.session_state.get("assistant_retrieval_error"):
        st.info("No research passages were retrieved for this question; the answer should state this limitation.")

    _render_pdf_selection("assistant")


def calculate_molecular_descriptors(smiles: str) -> dict[str, float | int]:
    """Calculate clearly labeled RDKit descriptors for a validated structure."""
    canonical = MolecularAbsorptionPredictor.canonicalize_smiles(smiles)
    molecule = Chem.MolFromSmiles(canonical)
    if molecule is None:
        raise InvalidSmilesError("RDKit could not parse this SMILES string.")
    return {
        "Molecular weight (g/mol)": round(float(Descriptors.MolWt(molecule)), 2),
        "XLogP": round(float(Crippen.MolLogP(molecule)), 2),
        "Topological polar surface area (Å²)": round(float(rdMolDescriptors.CalcTPSA(molecule)), 2),
        "Hydrogen bond donors": int(rdMolDescriptors.CalcNumHBD(molecule)),
        "Hydrogen bond acceptors": int(rdMolDescriptors.CalcNumHBA(molecule)),
        "Rotatable bonds": int(rdMolDescriptors.CalcNumRotatableBonds(molecule)),
    }


def build_prediction_explanation(
    predictions: dict[str, Any],
    descriptors: dict[str, Any] | None = None,
    compound_name: str | None = None,
    smiles: str = "",
) -> str:
    """Format the deterministic local explanation for display or other callers."""
    return generate_prediction_explanation(
        compound_name=compound_name,
        smiles=smiles,
        predictions=predictions,
        descriptors=descriptors,
    ).as_text()


def _clear_assistant_result() -> None:
    for key in (
        "assistant_result",
        "assistant_answer",
        "assistant_error",
        "assistant_error_type",
        "assistant_retrieval_error",
        "assistant_sources",
        "assistant_answer_mode",
        "assistant_custom_question",
        "assistant_suggested_question",
    ):
        st.session_state.pop(key, None)


def _clear_prediction_result() -> None:
    st.session_state.pop("oralabs_analysis", None)
    _clear_assistant_result()


def _predict_input_mode_changed() -> None:
    """Drop stale results when switching between library and manual inputs."""
    _clear_prediction_result()


def _predict_name_changed() -> None:
    previous_name = st.session_state.get("predict_last_name", "")
    current_name = str(st.session_state.get("predict_compound_name", "")).strip()
    if previous_name and current_name.casefold() != previous_name.casefold():
        # A changed label cannot keep the prior compound's structure attached.
        st.session_state["predict_smiles"] = ""
        st.session_state.pop("predict_verified_compound", None)
    st.session_state.pop("predict_ambiguous_name", None)
    st.session_state.pop("predict_ambiguous_cids", None)
    st.session_state.pop("predict_library_error", None)
    st.session_state["predict_last_name"] = current_name
    _clear_prediction_result()


def _predict_smiles_changed() -> None:
    current = str(st.session_state.get("predict_smiles", "")).strip()
    verified = st.session_state.get("predict_verified_compound")
    if isinstance(verified, dict) and current != verified.get("smiles"):
        st.session_state.pop("predict_verified_compound", None)
    _clear_prediction_result()


def _reset_predict_library() -> None:
    for key in (
        "predict_library_compound",
        "predict_last_library_name",
        "predict_verified_compound",
        "predict_ambiguous_name",
        "predict_ambiguous_cids",
        "predict_ambiguous_cid_choice",
    ):
        st.session_state.pop(key, None)
    _clear_prediction_result()


def _record_from_verified_selection(record: PubChemCompound, canonical: str) -> dict[str, Any]:
    return {
        "name": record.requested_name,
        "cid": record.cid,
        "smiles": canonical,
        "iupac_name": record.iupac_name,
        "inchikey": record.inchikey,
        "identity_verified": True,
    }


def _activate_library_compound(compound_name: str, prediction_page: Any) -> None:
    """Resolve a library choice and populate inputs, without running inference."""
    try:
        with st.spinner("Verifying the selected compound with PubChem and RDKit..."):
            verified = _resolve_library_compound(compound_name)
        st.session_state["predict_verified_compound"] = verified
        st.session_state["predict_compound_name"] = verified["name"]
        st.session_state["predict_smiles"] = verified["smiles"]
        st.session_state["predict_last_name"] = verified["name"]
        st.session_state.pop("predict_ambiguous_cids", None)
        st.session_state.pop("predict_ambiguous_name", None)
        _clear_prediction_result()
    except AmbiguousCompoundError as exc:
        st.session_state["predict_ambiguous_name"] = compound_name
        st.session_state["predict_ambiguous_cids"] = exc.cids
        st.session_state.pop("predict_verified_compound", None)
        st.session_state["predict_smiles"] = ""
        _clear_prediction_result()
    except (CompoundNotFoundError, PubChemServiceError, InvalidSmilesError) as exc:
        st.session_state["predict_library_error"] = str(exc)
        st.session_state.pop("predict_verified_compound", None)
        st.session_state["predict_smiles"] = ""
        _clear_prediction_result()


def _show_prediction_results(analysis: dict[str, Any]) -> None:
    predictions = analysis["result"]["predictions"]
    st.subheader("Prediction Results")
    columns = st.columns(3, gap="medium")
    for column, endpoint in zip(columns, ("HIA", "HOB20", "HOB50"), strict=True):
        result = predictions.get(endpoint)
        with column:
            if not isinstance(result, dict) or result.get("prediction") not in (0, 1):
                probability = None
                display, classification = "N/A", "Classification unavailable"
                color = "#8c99a7"
            else:
                probability = result.get("probability")
                if not isinstance(probability, (int, float)):
                    probability = None
                if probability is not None:
                    probability = max(0.0, min(1.0, float(probability)))
                    display = f"{probability * 100:.1f}%"
                else:
                    display = "N/A"
                classification = "Positive" if int(result["prediction"]) == 1 else "Negative"
                color = _prediction_ring_color(endpoint, int(result["prediction"]), probability)
            _result_circle(endpoint, display, classification, probability, color)

    if analysis["result"].get("consistency_warning"):
        st.warning(analysis["result"]["consistency_warning"])

    with st.container(border=True):
        st.subheader("Prediction Explanation")
        explanation = analysis.get("prediction_explanation")
        if (
            analysis.get("prediction_explanation_engine") != EXPLANATION_ENGINE_VERSION
            or not isinstance(explanation, str)
            or not explanation.strip()
        ):
            explanation = build_prediction_explanation(
                predictions,
                analysis.get("molecular_descriptors"),
                compound_name=analysis.get("compound_name"),
                smiles=analysis.get("canonical_smiles", ""),
            )
            analysis["prediction_explanation"] = explanation
            analysis["prediction_explanation_engine"] = EXPLANATION_ENGINE_VERSION
        st.markdown(explanation)


def _render_compound_assistant(analysis: dict[str, Any], research_page: Any) -> None:
    with st.container(border=True):
        _render_compound_assistant_content(analysis, research_page)


def _render_compound_assistant_content(analysis: dict[str, Any], research_page: Any) -> None:
    canonical = analysis["canonical_smiles"]
    current = st.session_state.get("assistant_result")
    if isinstance(current, dict) and current.get("compound_smiles") != canonical:
        _clear_assistant_result()
        current = None

    st.subheader("Ask About This Compound")
    st.caption("Explore its molecular properties, absorption predictions and scientific literature.")
    name = analysis.get("compound_name") or "this compound"
    suggestions = (
        f"Explain the absorption predictions for {name}.",
        f"What molecular properties may influence the absorption of {name}?",
        f"What does research report about the oral bioavailability of {name}?",
        f"What factors may contribute to limited intestinal absorption of {name}?",
        f"What formulation approaches have been investigated for {name}?",
        f"How should I interpret the HIA, HOB20 and HOB50 results for {name}?",
    )
    with st.expander("Suggested questions", expanded=False):
        selected_question = st.radio(
            "Choose a question",
            suggestions,
            key="assistant_suggested_question",
            label_visibility="collapsed",
        )
    custom_question = st.text_area(
        "Your question (optional)",
        placeholder="Ask any supported scientific question about this compound...",
        key="assistant_custom_question",
        height=96,
    )
    question = custom_question.strip() or selected_question
    if st.button("Get Answer", type="primary", key="assistant_submit"):
        st.session_state.pop("assistant_result", None)
        st.session_state.pop("assistant_answer", None)
        st.session_state.pop("assistant_sources", None)
        st.session_state.pop("assistant_error", None)
        try:
            with st.spinner("Searching relevant research and preparing an answer..."):
                index, index_warning = cached_research_index(
                    str(_index_directory()), DEFAULT_EMBEDDING_MODEL
                )
                qa = answer_with_research(
                    compound_name=str(analysis.get("compound_name") or ""),
                    compound_information=analysis.get("compound_information", {}),
                    question=question,
                    chemberta_predictions=analysis["result"]["predictions"],
                    api_key=(get_gemini_settings()[0] or ""),
                    gemini_model=get_gemini_settings()[1],
                    research_index=index,
                    research_index_error=None if index is not None else "ResearchIndexUnavailable",
                    molecular_descriptors=analysis.get("molecular_descriptors"),
                    structured_fallback=True,
                )
            st.session_state["assistant_result"] = {
                "compound_smiles": canonical,
                "question": question,
                "answer": qa.answer,
                "answer_mode": qa.answer_mode,
                "assistant_error": qa.assistant_error,
                "assistant_error_type": qa.assistant_error_type,
                "retrieval_error": qa.retrieval_error,
                "sources": _group_sources(qa.research_passages),
                "index_warning": index_warning,
            }
        except ValueError as exc:
            st.session_state["assistant_error"] = str(exc)

    result = st.session_state.get("assistant_result")
    if isinstance(result, dict) and result.get("compound_smiles") == canonical:
        if result.get("assistant_error"):
            if result.get("answer_mode") == "structured":
                st.info("Gemini was unavailable; this is a limited structured response based on supplied predictions or calculated properties.")
            else:
                st.warning(result["assistant_error"])
        if result.get("answer"):
            st.markdown("#### Answer")
            st.write(result["answer"])
        if result.get("retrieval_error"):
            st.info("No PDF evidence was retrieved for this question.")
        sources = result.get("sources", [])
        if sources:
            st.markdown("#### Relevant Research Sources")
            for position, source in enumerate(sources):
                filename = Path(source["filename"]).name
                page = source.get("page_number")
                label = f"{filename}, p. {page}" if page else filename
                with st.container(border=True):
                    st.markdown(f"**{source.get('title') or filename}** · {label}")
                    excerpt = " ".join(source.get("passages", []))
                    st.caption(excerpt[:320] + ("…" if len(excerpt) > 320 else ""))
                    if st.button("Open PDF page" if page else "Open PDF", key=f"compound_source_{position}"):
                        _set_pdf_page(filename, page)
                        st.rerun()
        elif not result.get("retrieval_error"):
            st.info("No relevant research passages were retrieved for this question.")
        if result.get("index_warning"):
            st.warning(result["index_warning"])

    st.markdown("#### Additional Research")
    research_count = len(cached_research_catalog())
    st.caption(f"Explore the full research library and its {research_count} PDFs.")
    st.page_link(research_page, label="Explore Research", icon=":material/library_books:")
    _render_pdf_selection("compound_assistant")


def render_predict_page(research_page: Any) -> None:
    st.markdown('<div class="mi-brand">MolInsight <span>ORALABS PREDICT</span></div>', unsafe_allow_html=True)
    st.title("Predict Oral Absorption")
    st.markdown(
        '<p class="mi-page-subtitle">Explore molecular absorption properties and understand the research behind your compound.</p>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.subheader("Choose your compound")
        input_mode = st.radio(
            "Input method",
            ("Compound library", "Enter manually"),
            horizontal=True,
            key="predict_input_mode",
            on_change=_predict_input_mode_changed,
            label_visibility="collapsed",
        )
        verified: dict[str, Any] | None = None
        if input_mode == "Compound library":
            category = st.selectbox(
                "Category",
                list(COMPOUND_LIBRARY),
                index=None,
                placeholder="Choose a category",
                key="predict_library_category",
                on_change=_reset_predict_library,
            )
            library_name = st.selectbox(
                "Compound",
                COMPOUND_LIBRARY.get(category, ()),
                index=None,
                placeholder="Choose a compound",
                key="predict_library_compound",
                disabled=category is None,
            )
            if library_name and library_name != st.session_state.get("predict_last_library_name"):
                st.session_state["predict_last_library_name"] = library_name
                st.session_state.pop("predict_library_error", None)
                _activate_library_compound(library_name, None)

            candidate_name = st.session_state.get("predict_ambiguous_name")
            candidate_cids = st.session_state.get("predict_ambiguous_cids", ())
            if candidate_name and candidate_cids:
                st.warning("PubChem found multiple possible identities. Choose the intended candidate to verify its structure.")
                candidate_cid = st.selectbox(
                    "Possible matches",
                    candidate_cids,
                    format_func=lambda cid: f"PubChem CID {cid}",
                    key="predict_ambiguous_cid_choice",
                )
                try:
                    candidate = cached_compound_lookup_by_cid(candidate_name, candidate_cid)
                    st.caption(f"Candidate IUPAC name: {candidate.iupac_name or 'Not reported'}")
                    if st.button("Use selected candidate", key="predict_use_candidate"):
                        canonical = MolecularAbsorptionPredictor.canonicalize_smiles(candidate.canonical_smiles)
                        verified = _record_from_verified_selection(candidate, canonical)
                        st.session_state["predict_verified_compound"] = verified
                        st.session_state["predict_compound_name"] = candidate_name
                        st.session_state["predict_smiles"] = canonical
                        st.session_state["predict_last_name"] = candidate_name
                        st.session_state.pop("predict_ambiguous_cids", None)
                        st.session_state.pop("predict_ambiguous_name", None)
                        _clear_prediction_result()
                        st.rerun()
                except (PubChemServiceError, InvalidSmilesError) as exc:
                    st.warning(f"The selected PubChem candidate could not be verified: {exc}")

            library_error = st.session_state.pop("predict_library_error", None)
            if library_error:
                st.warning(f"This library entry could not be verified: {library_error}")
            verified = st.session_state.get("predict_verified_compound")
            if isinstance(verified, dict):
                compound_name = str(verified.get("name") or library_name or "")
                smiles = str(verified.get("smiles") or "")
                st.caption("Library compounds are verified through PubChem before prediction.")
                st.success(
                    f"Verified identity: {compound_name} · PubChem CID {verified.get('cid')} · "
                    f"SMILES: `{smiles}`"
                )
            else:
                compound_name, smiles = "", ""
                st.caption("Choose a library entry to load its verified structure. Prediction runs only when you submit.")
        else:
            first, second = st.columns(2, gap="medium")
            with first:
                compound_name = st.text_input(
                    "Compound name",
                    placeholder="Enter a compound name",
                    key="predict_compound_name",
                    on_change=_predict_name_changed,
                )
            with second:
                smiles = st.text_input(
                    "Molecular SMILES",
                    placeholder="Enter or paste a SMILES string",
                    key="predict_smiles",
                    on_change=_predict_smiles_changed,
                )
            verified = st.session_state.get("predict_verified_compound")
            if isinstance(verified, dict) and verified.get("smiles") == smiles.strip():
                st.success(
                    f"Verified identity: {verified.get('name')} · PubChem CID {verified.get('cid')} · "
                    f"SMILES: `{verified.get('smiles')}`"
                )
            elif compound_name.strip() and not smiles.strip():
                st.caption("If PubChem cannot resolve the name, enter a SMILES manually; RDKit validates the structure.")

        submitted = st.button("Predict Absorption", type="primary", key="predict_absorption_submit")
        st.caption("Computational estimates for research use; not experimental measurements or medical advice.")

    if submitted:
        _clear_prediction_result()
        st.session_state.pop("predict_ambiguous_name", None)
        st.session_state.pop("predict_ambiguous_cids", None)
        if not compound_name.strip() and not smiles.strip():
            st.error("Enter a compound name or a molecular SMILES before predicting.")
        else:
            try:
                compound_record: PubChemCompound | None = None
                canonical = ""
                if smiles.strip():
                    canonical = MolecularAbsorptionPredictor.canonicalize_smiles(smiles)
                    if (
                        isinstance(verified, dict)
                        and verified.get("smiles") == canonical
                        and verified.get("cid")
                    ):
                        compound_record = PubChemCompound(
                            requested_name=str(verified.get("name") or compound_name.strip()),
                            cid=int(verified["cid"]),
                            canonical_smiles=canonical,
                            iupac_name=verified.get("iupac_name"),
                            inchikey=verified.get("inchikey"),
                        )
                else:
                    try:
                        compound_record = cached_compound_lookup(compound_name.strip())
                    except AmbiguousCompoundError as exc:
                        st.session_state["predict_ambiguous_name"] = compound_name.strip()
                        st.session_state["predict_ambiguous_cids"] = exc.cids
                        st.warning("PubChem found multiple possible identities. Choose a candidate above.")
                        st.stop()
                    canonical = MolecularAbsorptionPredictor.canonicalize_smiles(
                        compound_record.canonical_smiles
                    )

                prediction_output = cached_prediction(str(default_model_dir()), canonical)
                prediction_result = _prediction_result_for_display(prediction_output)
                descriptors = calculate_molecular_descriptors(canonical)
                inchikey = (
                    compound_record.inchikey
                    if compound_record and compound_record.inchikey
                    else cached_inchikey(canonical)
                )
                try:
                    chembl_record = cached_chembl_lookup(inchikey)
                    chembl_status = "found"
                except ChemblNotFoundError:
                    chembl_record, chembl_status = None, "not_found"
                except (ChemblAmbiguousError, ChemblServiceError, ValueError):
                    chembl_record, chembl_status = None, "unavailable"

                verified_identity = compound_record is not None
                display_name = (
                    compound_record.iupac_name or compound_record.requested_name
                    if compound_record
                    else compound_name.strip()
                )
                compound_info: dict[str, Any] = {
                    "RDKit calculated descriptors": descriptors,
                    "Structure identity": (
                        "Verified through PubChem."
                        if verified_identity
                        else "User-provided SMILES; compound label is not verified through PubChem."
                    ),
                }
                if compound_record:
                    compound_info["PubChem"] = {
                        "CID": compound_record.cid,
                        "IUPAC name": compound_record.iupac_name,
                        "canonical SMILES": canonical,
                        "InChIKey": compound_record.inchikey,
                        "record URL": f"https://pubchem.ncbi.nlm.nih.gov/compound/{compound_record.cid}",
                    }
                if chembl_record:
                    compound_info["ChEMBL"] = {
                        "molecule ID": chembl_record.chembl_id,
                        "preferred name": chembl_record.preferred_name,
                        "molecular formula": chembl_record.molecular_formula,
                        "molecular weight": chembl_record.molecular_weight,
                        "reported properties": chembl_record.reported_properties,
                        "record URL": chembl_record.record_url,
                    }

                st.session_state["oralabs_analysis"] = {
                    "compound_name": display_name,
                    "identity_verified": verified_identity,
                    "canonical_smiles": canonical,
                    "input_name": compound_name.strip(),
                    "input_smiles": smiles.strip(),
                    "compound_record": _serialize_compound(compound_record, canonical) if compound_record else None,
                    "compound_information": compound_info,
                    "molecular_descriptors": descriptors,
                    "chembl_record": chembl_record,
                    "chembl_status": chembl_status,
                    "result": prediction_result,
                }
                explanation = generate_prediction_explanation(
                    compound_name=display_name,
                    smiles=canonical,
                    predictions=prediction_result["predictions"],
                    descriptors=descriptors,
                )
                st.session_state["oralabs_analysis"]["prediction_explanation"] = explanation.as_text()
                st.session_state["oralabs_analysis"]["prediction_explanation_engine"] = EXPLANATION_ENGINE_VERSION
                if compound_record:
                    st.session_state["predict_verified_compound"] = _record_from_verified_selection(compound_record, canonical)
            except CompoundNotFoundError:
                st.error("PubChem could not resolve that name. Enter a SMILES manually to use the compound label without claiming a verified identity.")
            except (PubChemServiceError, InvalidSmilesError) as exc:
                st.error(str(exc))
            except FileNotFoundError:
                st.error("The bundled ChemBERTa V1 model files are unavailable.")

    analysis = st.session_state.get("oralabs_analysis")
    if not isinstance(analysis, dict):
        return
    if (
        analysis.get("input_name") != compound_name.strip()
        or analysis.get("input_smiles") != smiles.strip()
    ):
        st.info("The displayed results belong to the last submitted compound. Submit this input to run a new prediction.")
        return

    if analysis.get("compound_name"):
        label_note = " · user label, identity unverified" if not analysis.get("identity_verified") else " · verified PubChem identity"
        st.caption(f"Compound: {analysis['compound_name']}{label_note}")
    st.caption(f"Canonical SMILES: `{analysis['canonical_smiles']}`")
    _show_prediction_results(analysis)
    _render_compound_assistant(analysis, research_page)
