from __future__ import annotations

import streamlit as st

from chembl import (
    ChemblAmbiguousError,
    ChemblClient,
    ChemblNotFoundError,
    ChemblServiceError,
    inchikey_from_smiles,
)
from inference import InvalidSmilesError, MolecularAbsorptionPredictor, default_model_dir
from pubchem import (
    AmbiguousCompoundError,
    CompoundNotFoundError,
    PubChemClient,
    PubChemServiceError,
)


st.set_page_config(page_title="Oralabs Predict", layout="centered")
st.markdown(
    """
    <style>
    .block-container { max-width: 960px; padding-top: 2.8rem; padding-bottom: 3rem; }
    h1, h2 { color: #142B4A; letter-spacing: -0.025em; }
    h1 { font-size: 2.25rem; font-weight: 700; }
    div[data-testid="stTextInput"] label { color: #233B5D; font-weight: 600; }
    div.stButton > button {
        width: 100%;
        min-height: 3rem;
        border-radius: 0.65rem;
        border: 0;
        background: #2563EB;
        color: #FFFFFF;
        font-weight: 650;
    }
    div.stButton > button:hover { background: #1D4ED8; color: #FFFFFF; }
    .prediction-card {
        min-height: 170px;
        padding: 1.25rem 1.1rem;
        border: 1px solid transparent;
        border-radius: 0.9rem;
        text-align: center;
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: 0.55rem;
    }
    .prediction-endpoint { color: #193354; font-size: 1.05rem; font-weight: 700; }
    .prediction-probability { color: #142B4A; font-size: 2rem; font-weight: 750; }
    .prediction-class { color: #29415F; font-size: 0.98rem; font-weight: 600; }
    .hia-card { background: #EEF6FF; border-color: #D7E9FC; }
    .hob20-card { background: #EFF8F1; border-color: #D8ECDD; }
    .hob50-card { background: #FFF7E8; border-color: #F4E5C5; }
    @media (max-width: 640px) {
        div[data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] { min-width: 100%; }
        .prediction-card { min-height: 130px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Oralabs Predict")
st.markdown("Predict and analyze the oral absorption properties of chemical compounds.")


@st.cache_resource(show_spinner=False)
def load_predictor(model_path: str) -> MolecularAbsorptionPredictor:
    return MolecularAbsorptionPredictor(model_dir=model_path)


@st.cache_data(max_entries=128, show_spinner=False)
def cached_prediction(model_path: str, smiles: str) -> dict:
    predictor = load_predictor(model_path)
    return predictor.predict(smiles)


@st.cache_data(ttl=3600, max_entries=128, show_spinner=False)
def cached_compound_lookup(name: str):
    return PubChemClient().lookup_compound(name)


@st.cache_data(ttl=3600, max_entries=256, show_spinner=False)
def cached_inchikey(smiles: str) -> str:
    return inchikey_from_smiles(smiles)


@st.cache_data(ttl=3600, max_entries=128, show_spinner=False)
def cached_chembl_lookup(inchikey: str):
    return ChemblClient().lookup_by_inchikey(inchikey)


model_dir = default_model_dir()
input_mode = st.radio("Input type", ("Compound Name", "SMILES"), index=1, horizontal=True)
input_label = "Enter compound name" if input_mode == "Compound Name" else "Enter compound SMILES"
input_placeholder = "e.g. Aspirin" if input_mode == "Compound Name" else "Enter a compound SMILES string"
compound_input = st.text_input(input_label, placeholder=input_placeholder, key="compound_input")

if st.button("Predict absorption", type="primary"):
    identified_compound = None
    chembl_compound = None
    chembl_status = "not_checked"
    chembl_ambiguous_ids: tuple[str, ...] = ()
    try:
        if input_mode == "Compound Name":
            identified_compound = cached_compound_lookup(compound_input.strip())
            smiles = identified_compound.canonical_smiles
        else:
            smiles = compound_input.strip()
        result = cached_prediction(str(model_dir), smiles)
        try:
            structure_key = identified_compound.inchikey if identified_compound else None
            if not structure_key:
                structure_key = cached_inchikey(smiles)
            chembl_compound = cached_chembl_lookup(structure_key)
            chembl_status = "found"
        except ChemblNotFoundError:
            chembl_status = "not_found"
        except ChemblAmbiguousError as exc:
            chembl_status = "ambiguous"
            chembl_ambiguous_ids = exc.chembl_ids
        except ChemblServiceError:
            chembl_status = "unavailable"
        except ValueError:
            chembl_status = "unavailable"
    except InvalidSmilesError as exc:
        st.error(str(exc))
    except CompoundNotFoundError as exc:
        st.error(str(exc))
    except AmbiguousCompoundError as exc:
        preview = ", ".join(map(str, exc.cids[:5]))
        remaining = len(exc.cids) - 5
        suffix = f" and {remaining} more" if remaining > 0 else ""
        st.error(
            f"That name matches multiple PubChem compounds (CIDs: {preview}{suffix}). "
            "Add more detail to identify one compound."
        )
    except PubChemServiceError:
        st.error("PubChem could not be reached or returned an unavailable response. Please try again.")
    except FileNotFoundError:
        st.error("The prediction model is unavailable. Please check that the model files are installed.")
    except Exception:
        st.error("The prediction could not be completed. Please try again.")
    else:
        if identified_compound is not None:
            st.write(
                f"Identified compound: {identified_compound.requested_name} "
                f"(PubChem CID {identified_compound.cid})"
            )
        st.subheader("Prediction Results")
        columns = st.columns(3, gap="medium")
        for column, endpoint, style in zip(
            columns,
            ("HIA", "HOB20", "HOB50"),
            ("hia-card", "hob20-card", "hob50-card"),
            strict=True,
        ):
            prediction = result["predictions"][endpoint]
            classification = "Positive" if prediction["prediction"] else "Negative"
            column.markdown(
                f"""
                <div class="prediction-card {style}">
                    <div class="prediction-endpoint">{endpoint}</div>
                    <div class="prediction-probability">{prediction['probability'] * 100:.1f}%</div>
                    <div class="prediction-class">{classification}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if result["consistency_warning"]:
            st.warning(result["consistency_warning"])

        with st.expander("Compound information"):
            if identified_compound is not None:
                st.markdown("**PubChem**")
                st.markdown(
                    f"[PubChem CID {identified_compound.cid}]("
                    f"https://pubchem.ncbi.nlm.nih.gov/compound/{identified_compound.cid})"
                )
                if identified_compound.iupac_name:
                    st.write(f"IUPAC name: {identified_compound.iupac_name}")
                else:
                    st.write("IUPAC name: Not reported by PubChem.")

            st.markdown("**ChEMBL**")
            if chembl_status == "found" and chembl_compound is not None:
                st.markdown(
                    f"[ChEMBL molecule {chembl_compound.chembl_id}]({chembl_compound.record_url})"
                )
                st.write(
                    f"Preferred name: {chembl_compound.preferred_name or 'Not reported by ChEMBL.'}"
                )
                st.write(
                    f"Molecular formula: {chembl_compound.molecular_formula or 'Not reported by ChEMBL.'}"
                )
                weight = (
                    f"{chembl_compound.molecular_weight:g} g/mol"
                    if chembl_compound.molecular_weight is not None
                    else "Not reported by ChEMBL."
                )
                st.write(f"Molecular weight: {weight}")
                if chembl_compound.reported_properties:
                    st.markdown("Reported properties:")
                    for property_name, value in sorted(chembl_compound.reported_properties.items()):
                        st.write(f"- {property_name.replace('_', ' ')}: {value}")
            elif chembl_status == "not_found":
                st.write("No ChEMBL record was found for this structure.")
            elif chembl_status == "ambiguous":
                st.write(
                    "Multiple exact ChEMBL records matched this structure; no record was selected. "
                    f"Matches: {', '.join(chembl_ambiguous_ids)}."
                )
            elif chembl_status == "unavailable":
                st.write("ChEMBL is currently unavailable; compound information could not be retrieved.")
