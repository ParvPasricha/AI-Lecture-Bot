#!/usr/bin/env python3
"""
AI Lecture Bot with Web-Enhanced Question Generation & .env Support
------------------------------------------------------------------
Dependencies (pip):
    torch
    transformers
    sentencepiece
    flask
    google-search-results
    python-dotenv

Usage:
1. Create a `.env` file in the project root with:
    SERPAPI_API_KEY=your_serpapi_key_here
2. Ensure `.env` is listed in your `.gitignore`.
3. Install dependencies: pip install -r requirements.txt
4. Run: python app.py
"""

import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from serpapi import GoogleSearch

# ─── Step 1: Load environment variables from .env ───────────────────
load_dotenv()  # reads .env and sets os.environ entries
# Quick check (remove in production)
print("🔑 SerpAPI key loaded:", bool(os.getenv("SERPAPI_API_KEY")))

# ─── Step 2: Initialize Flask app ───────────────────────────────────
app = Flask(__name__)

# ─── Step 3: Load NLP Models ────────────────────────────────────────
# Summarization model (BART-large) on CPU
summarizer = pipeline(
    task="summarization",
    model="facebook/bart-large-cnn",
    device_map="cpu"
)

# Question-generation model (T5) with slow tokenizer to avoid extra deps
tokenizer_qg = AutoTokenizer.from_pretrained(
    "valhalla/t5-base-qg-hl",
    use_fast=False
)
model_qg = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-base-qg-hl")


# ─── Step 4: Core Helper Functions ───────────────────────────────────

def summarize_text(text: str) -> str:
    """
    Summarize the lecture text into ~200 tokens.
    """
    result = summarizer(text, max_length=200, min_length=50, do_sample=False)
    return result[0]["summary_text"]


def break_into_topics(summary: str) -> list[str]:
    """
    Split the summary on '. ' so each sentence is a topic.
    """
    return [s.strip() for s in summary.split(". ") if s.strip()]


def fetch_web_info(topic: str, num_results: int = 3) -> str:
    """
    If SERPAPI_API_KEY is set, query Google via SerpAPI for the topic
    and return the top `num_results` snippets concatenated.
    """
    api_key = os.getenv("SERPAPI_API_KEY", "")
    if not api_key:
        return ""  # no key → skip web enrichment

    search = GoogleSearch({
        "engine":  "google",
        "q":       topic,
        "num":     num_results,
        "api_key": api_key
    })
    data = search.get_dict()
    snippets = [item.get("snippet", "") for item in data.get("organic_results", [])]
    return " ".join(snippets)


def generate_questions(topics: list[str]) -> list[str]:
    """
    For each topic, optionally enrich with web info, then generate
    an exam-style question using T5.
    """
    questions = []
    for topic in topics:
        web_info = fetch_web_info(topic)
        prompt = (
            f"Based on: {topic}. "
            f"Additional info: {web_info} "
            "Generate an exam question:</s>"
        )
        input_ids = tokenizer_qg.encode(prompt, return_tensors="pt")
        output_ids = model_qg.generate(
            input_ids,
            max_length=64,
            num_beams=4,
            early_stopping=True
        )
        question = tokenizer_qg.decode(output_ids[0], skip_special_tokens=True)
        questions.append(question)
    return questions


# ─── Step 5: Flask API Endpoint ─────────────────────────────────────

@app.route("/summarize", methods=["POST"])
def summarize_endpoint():
    """
    POST /summarize
    Request JSON: { "text": "<lecture text>" }
    Response JSON:
    {
        "summary":    str,
        "key_topics": [str, ...],
        "questions":  [str, ...]
    }
    Returns 400 if "text" is missing or empty.
    """
    payload = request.get_json(force=True)
    lecture_text = payload.get("text", "").strip()
    if not lecture_text:
        return jsonify({"error": "No 'text' supplied"}), 400

    # 1) Summarize the lecture
    summary = summarize_text(lecture_text)
    # 2) Break summary into topics
    topics = break_into_topics(summary)
    # 3) Generate questions enriched with web snippets
    questions = generate_questions(topics)

    return jsonify({
        "summary":    summary,
        "key_topics": topics,
        "questions":  questions
    })


# ─── Step 6: Run the App ─────────────────────────────────────────────

if __name__ == "__main__":
    # host="0.0.0.0" allows external connections; debug=True for dev
    app.run(host="0.0.0.0", port=5000, debug=True)
