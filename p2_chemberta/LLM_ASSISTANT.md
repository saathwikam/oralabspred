# Gemini compound Q&A

The Streamlit app includes an optional Gemini Q&A section after a valid compound prediction. The independent `llm_assistant.py` module accepts a compound name, source-labeled PubChem/ChEMBL information, the question, and optionally existing ChemBERTa predictions. It does not import or call the compound lookup or model modules.

The assistant distinguishes supplied source records from general scientific explanations, says when the records do not provide enough information, and must not invent citations. ChemBERTa predictions are passed only for questions that mention an endpoint, prediction, classification, or probability. The prompt prohibits the LLM from creating or changing model predictions and asks it to explain supplied outputs with their limitations. Research-paper retrieval and document citations are not part of this feature.

## Configure the API key

Set `GEMINI_API_KEY` in the environment, or add it to Streamlit secrets. For local development, `.streamlit/secrets.toml` can contain:

```toml
GEMINI_API_KEY = "your-key"
GEMINI_MODEL = "gemini-3.8-flash"
```

The secrets file is ignored by Git. In Render, add `GEMINI_API_KEY` and optionally `GEMINI_MODEL` as environment variables in the service settings. Never commit a real key. Environment variables take precedence over Streamlit secrets.

The default model is `gemini-3.8-flash`. Set `GEMINI_MODEL` to another model ID supported by your Gemini API project to change it.

## Run

Install the project dependencies and launch the existing app from the repository root:

```powershell
py -3.10 -m pip install -r requirements.txt
py -3.10 -m streamlit run p2_chemberta/streamlit_app.py
```

After entering a compound and selecting **Predict absorption**, open **Ask a question**, enter a question, and submit it. Missing keys, quota/rate limits, network failures, and other API errors are reported in the Q&A section; they do not change or block ChemBERTa predictions.

Run the mocked API unit tests without making Gemini API calls:

```powershell
py -3.10 -m unittest tests.test_llm_assistant -v
```
