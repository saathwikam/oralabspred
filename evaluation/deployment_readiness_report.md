# Oralabs Predict deployment readiness audit

**Audit scope:** Render deployment configuration, V1 inference artifacts and paths, CPU dependencies, optional Gemini/RAG behavior, Git exclusions, and memory feasibility. No model, UI, data, or prediction code was changed.

## Summary

| Area | Status | Finding |
|---|---|---|
| Render startup | Ready | `render.yaml` runs Streamlit on `0.0.0.0:$PORT`. |
| V1 model bundle | Ready | The root `best_model/` contains all six required checkpoint, encoder, and tokenizer files. |
| CPU inference | Ready, resource-tight | CPU-only PyTorch is pinned and inference defaults to CPU. The prior local post-prediction measurement was 431 MiB. |
| Gemini configuration | Ready for deployment configuration | The code accepts `GEMINI_API_KEY` from environment or Streamlit secrets. Configure it privately in Render; no key is required for predictions. |
| RAG | Not provisioned | Neither the default `rag_store/` index nor `research_documents/` is present. Research-grounded answers are unavailable until an index is supplied. |
| Optional-service degradation | Ready by code inspection | Prediction is performed independently of Q&A. RAG load/retrieval and Gemini configuration/API errors are caught and surfaced as user messages. |
| Git hygiene | Mostly ready; working tree needs review | Ignore rules cover secrets, local documents/indexes, data and outputs. No tracked paths matching secrets, `.env`, `research_documents`, or `rag_store` were returned by the available path scan. The Git status command itself reported a worktree error, so staging status could not be fully confirmed in this audit. |
| Render Free for full app | Not reliably feasible | The V1 process is already close to the 512 MB limit; loading the RAG embedding model in the same service is likely to exceed it. |

## Findings

### 1. Render configuration and startup

`render.yaml` declares a Free Python web service, installs `requirements.txt`, and starts:

```sh
streamlit run p2_chemberta/streamlit_app.py --server.address 0.0.0.0 --server.port $PORT
```

This is Render-compatible: it binds externally and uses Render''s assigned port. The configured Python version is 3.10.11.

### 2. Model and tokenizer paths

The root `best_model/` contains exactly the expected inference bundle:

- `model_config.json`
- `model_state_dict.pt`
- `encoder/config.json`
- `encoder/model.safetensors`
- `tokenizer/tokenizer_config.json`
- `tokenizer/tokenizer.json`

The inference module resolves `ORAL_ABSORPTION_MODEL_DIR` when set and otherwise uses the project-root `best_model/`. The encoder and state dictionary load on CPU (`map_location` follows the requested device); the loaded model is placed in evaluation mode. Streamlit caches the predictor as a resource, avoiding a model reload on every rerun. Existing verification reported all five inference tests passing against the actual checkpoint; no fresh rerun was possible in this audit shell (see verification limitation below).

### 3. CPU dependencies

The pinned requirements include Streamlit, CPU-only PyTorch, Transformers, RDKit, NumPy, Safetensors, the Gemini SDK, PyMuPDF, Sentence Transformers, and FAISS CPU. These cover the inference, compound lookup, Gemini, and RAG imports seen in the application; PubChem/ChEMBL HTTP calls use the Python standard library.

The active audit shell only exposes MSYS Python without PyTorch and has no registered Python installations. Thus I could not reinstall or freshly import-test the pinned environment here. The existing local Python 3.10 inference verification is the available compatibility evidence; Render build compatibility remains to be confirmed by an actual build.

### 4. Memory feasibility

A prior local measurement recorded **431 MiB after a ChemBERTa prediction**. Against Render Free''s **512 MB** memory limit, this leaves only about **81 MiB** of headroom before accounting for deployment/runtime variation. Render Free also provides limited CPU. The V1-only service may start and predict, but this margin is too small to call reliable.

The RAG workflow lazily loads `sentence-transformers/all-MiniLM-L6-v2` during retrieval, in the same process as the cached ChemBERTa model. With a RAG index present, that second model and its runtime memory make exceeding the limit likely. Peak memory has not been measured on Render, so this is a risk estimate, not an observed Render failure. Render documents the Free instance limit and its filesystem behavior in its [Free plan documentation](https://render.com/docs/free) and [compute plan specifications](https://render.com/docs/compute-plans).

### 5. Gemini secrets and graceful degradation

The assistant reads `GEMINI_API_KEY` from the environment first and can fall back to Streamlit secrets. The Render manifest intentionally contains no credential. Set the key in Render''s private environment-variable settings if Gemini Q&A is wanted; never commit it.

The prediction flow executes before optional Q&A. Missing Gemini configuration and Gemini/API errors are handled in the Q&A path, while RAG loading/retrieval failures are caught and Q&A can proceed without retrieved evidence. Missing RAG evidence is therefore a limitation on research answers, not a prerequisite for HIA/HOB20/HOB50 prediction. General compound lookup failures also have dedicated handling in the app.

### 6. RAG index and documents

The code''s default index is project-root `rag_store/`, overridable with `ORALABS_RAG_INDEX_DIR`. The loader expects the FAISS index and associated chunk/manifest metadata there. At audit time, **`rag_store/` and `research_documents/` are both absent**. The application can degrade to Gemini without retrieved research, but cannot provide document-backed research responses until an index is provisioned.

Render Free''s filesystem is ephemeral, so an index created manually in a running instance is not durable across restarts/redeploys. The [Render Free plan documentation](https://render.com/docs/free) describes this limitation. Provision the index as part of the deployable application artifact or another zero-cost durable workflow before advertising RAG functionality.

### 7. Git exclusions and repository state

`.gitignore` excludes Streamlit secrets, environment files, credential-like files, Kaggle credentials, local research documents, `rag_store/`, processed datasets, training outputs, and temporary files. It does not ignore the root `best_model/`, which is required for deployment. The inspected tracked-path listing had no matching secret, `.env`, research-document, or RAG-index paths.

A complete staging check remains necessary before any push: `git status --short` in this checkout returned “fatal: this operation must be run in a work tree” despite `git rev-parse --show-toplevel` resolving the repository. No staging or repository changes were made in this audit.

## Minimal fixes required

No source-code fix is essential for the current V1 prediction path or for graceful handling of absent Gemini/RAG. Before a Render launch:

1. Configure `GEMINI_API_KEY` in Render''s private settings only if Gemini Q&A is enabled.
2. Decide whether research-grounded Q&A is part of launch. If yes, provision the existing RAG index at the configured path and validate memory; otherwise clearly treat RAG-backed answers as unavailable.
3. Resolve the Git worktree/status issue and review the exact deployment commit contents.
4. Run the pinned dependency install and app smoke test in the deployment build environment.
5. Measure peak memory with both models loaded before relying on Render Free.

No files other than this audit report were modified.

## Verification limitation

The earlier actual-checkpoint verification reported five passing inference tests. During this audit, a fresh attempt could not run because `Python310/python.exe` was not present in the checkout, `py -0p` reported no registered Python installations, and the available MSYS Python lacks PyTorch. No test result is claimed for this audit session.

## Final assessment

**Ready components:** Render command/port, portable V1 artifact path, six model files, CPU-oriented inference configuration, secret lookup design, and optional Gemini/RAG failure isolation.

**Missing components:** A deployable RAG index and source research documents; a working local test environment in this audit shell; a verified clean staging state.

**Critical deployment blockers:** None identified for V1-only predictions. The missing RAG index blocks research-grounded retrieval. The 431 MiB local V1 memory measurement leaves too little headroom to consider the combined ChemBERTa + RAG service reliable on Render Free.

**Render Free feasibility:** V1-only may be feasible but is borderline and needs a cold-start/peak-memory check. The full current application with ChemBERTa plus on-demand Sentence Transformer retrieval is **not realistically reliable at 512 MB** without measured evidence to the contrary. No deployment was attempted.

