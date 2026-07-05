# Internal Knowledge Assistant (AI Chatbot)

A Flask REST API + simple web UI that answers questions about company
documents (safety, procurement, onboarding) by retrieving the most
relevant passage (TF-IDF) and passing it as context to an LLM.

## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
export ANTHROPIC_API_KEY=your_key_here   # optional
python app.py
```
Open http://localhost:5000, or call the API directly:
```bash
curl -X POST http://localhost:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What safety gear is required on site?"}'
```

- With `ANTHROPIC_API_KEY` set: the LLM generates a natural-language
  answer grounded in the retrieved passage.
- Without it: the API returns the raw retrieved passage(s) so the
  retrieval logic can be tested without any API key.

## What this demonstrates
- TF-IDF document retrieval (scikit-learn) as a lightweight RAG step
- LLM API integration with prompt grounding to reduce hallucination
- Flask REST API + minimal front end
- Graceful offline fallback and error handling
