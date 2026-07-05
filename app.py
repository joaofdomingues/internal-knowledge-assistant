"""
Internal Knowledge Assistant
-----------------------------
A small Flask app + REST API that answers questions about company
documentation (safety procedures, procurement policy, onboarding, etc.)
by retrieving the most relevant text chunks (TF-IDF) and passing them
as context to an LLM.

Run:
    python app.py
Then open http://localhost:5000 in a browser, or POST to /api/ask.

If ANTHROPIC_API_KEY is not set, the app runs in offline mode and
simply returns the most relevant retrieved passage(s) instead of an
LLM-generated answer, so the whole flow can still be tested.
"""
import os
import json
import requests
from flask import Flask, request, jsonify, render_template
from retriever import KnowledgeBase

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-5"

BASE_DIR = os.path.dirname(__file__)
kb = KnowledgeBase(os.path.join(BASE_DIR, "knowledge_base"))

app = Flask(__name__)

ANSWER_PROMPT = """Answer the user's question using ONLY the context below.
If the answer isn't in the context, say you don't have that information.
Be concise (2-4 sentences).

Context:
{context}

Question: {question}
"""


def ask_llm(question: str, context: str, api_key: str) -> str:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": MODEL,
        "max_tokens": 300,
        "messages": [{"role": "user", "content": ANSWER_PROMPT.format(context=context, question=question)}],
    }
    resp = requests.post(ANTHROPIC_API_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/ask", methods=["POST"])
def api_ask():
    payload = request.get_json(force=True)
    question = (payload or {}).get("question", "").strip()
    if not question:
        return jsonify({"error": "Missing 'question' field"}), 400

    results = kb.search(question, top_k=2)
    context = "\n\n".join(r["text"] for r in results) or "No relevant documentation found."
    sources = [r["source"] for r in results]

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            answer = ask_llm(question, context, api_key)
        except Exception as e:
            answer = f"(LLM call failed: {e}) Closest matching passage:\n{context}"
    else:
        answer = f"[OFFLINE MODE - no API key set] Closest matching passage(s):\n{context}"

    return jsonify({"question": question, "answer": answer, "sources": sources})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
