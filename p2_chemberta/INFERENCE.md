# Trained model inference

The Streamlit app supports direct SMILES and compound-name input. The name lookup is isolated in `pubchem.py`, which uses PubChem PUG REST to resolve one CID and retrieve its SMILES, IUPAC name and InChIKey. ChEMBL retrieval is isolated in `chembl.py` and uses the exact standard InChIKey, supplied by PubChem or computed from a validated SMILES, to find molecule records. Unknown names, ambiguous matches and service failures are handled without guessing. Retrieved SMILES are sent through the existing RDKit validation and inference path.

PubChem and ChEMBL records are shown separately in the optional compound-information expander. Missing fields are labeled as not reported by their source; service failures and absent ChEMBL records are reported separately. The record IDs and source links are kept distinct.

The app loads the recovered Kaggle `best_model` directory once per Streamlit process using `st.cache_resource`. Identical SMILES inputs reuse cached predictions, avoiding repeated RDKit validation, tokenization and model inference. PubChem lookups are cached for one hour. By default the model directory is beside the project folder at:

`<project-parent>/AI_Based_Oral_Absorption_Prediction_System_Kaggle_Bundle/outputs/AI_Based_Oral_Absorption_Prediction_System/best_model`

To use a different location, set `ORAL_ABSORPTION_MODEL_DIR` to the `best_model` directory. The loaded `model_config.json` supplies the endpoint order, maximum sequence length and 0.5 prediction threshold. The app validates and canonicalizes SMILES with RDKit, then reports each endpoint's probability and binary prediction. HOB20 and HOB50 endpoint definitions are shown separately from the classification probability cutoff. A HOB20/HOB50 consistency warning never changes those predictions. `.streamlit/config.toml` sets the professional light theme.

Run from the project root:

```powershell
py -3.10 -m pip install streamlit
py -3.10 -m streamlit run p2_chemberta/streamlit_app.py
```

Run the focused inference and mocked PubChem/ChEMBL checks with the recovered model artifacts available at the default path:

```powershell
py -3.10 -m unittest tests.test_inference tests.test_pubchem tests.test_chembl -v
```
