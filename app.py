#!/usr/bin/env python3
"""
AI Lecture Summarizer & Question Generator Bot
------------------------------------------------
Dependencies (install via pip):
    torch transformers sentencepiece flask
"""

from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

# --- Model Initialization -----------------------------------------
# Summarization model: BART-large-cnn (runs on CPU)
summarizer = pipeline(
    task="summarization",
    model="facebook/bart-large-cnn",
    device_map="cpu"      # ensure CPU usage
)

# Question-generation model: T5 with slow tokenizer (avoids blobfile errors)
tokenizer_qg = AutoTokenizer.from_pretrained(
    "valhalla/t5-base-qg-hl",
    use_fast=False        # disable fast tokenizers to skip extra deps
)
model_qg = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-base-qg-hl")


# --- Helper Functions --------------------------------------------
def summarize_text(text: str) -> str:
    """
    Summarize the input lecture text into a shorter paragraph.
    Returns a single summary string.
    """
    result = summarizer(
        text,
        max_length=200,
        min_length=50,
        do_sample=False
    )
    return result[0]["summary_text"]


def break_into_topics(summary: str) -> list[str]:
    """
    Split the summary into individual topic sentences.
    Each sentence becomes one 'topic'.
    """
    # split on period + space, filter out any empty strings
    return [s.strip() for s in summary.split(". ") if s.strip()]


def generate_questions(topics: list[str]) -> list[str]:
    """
    For each topic sentence, generate one exam-style question using T5.
    Returns a list of question strings.
    """
    questions = []
    for topic in topics:
        prompt = f"generate question: {topic} </s>"
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


# --- Flask Endpoint ----------------------------------------------
@app.route("/summarize", methods=["POST"])
def summarize_endpoint():
    """
    POST /summarize
    Expects JSON body: { "text": "<lecture content>" }
    Returns JSON:
    {
        "summary": str,
        "key_topics": [str],
        "questions": [str]
    }
    """
    payload = request.get_json(force=True)
    lecture_text = payload.get("text", "").strip()

    if not lecture_text:
        return jsonify({"error": "No 'text' supplied"}), 400

    # 1) Summarize
    summary = summarize_text(lecture_text)
    # 2) Extract topics
    topics = break_into_topics(summary)
    # 3) Generate questions
    questions = generate_questions(topics)

    # Bundle and send response
    return jsonify({
        "summary": summary,
        "key_topics": topics,
        "questions": questions
    })


# --- App Runner ---------------------------------------------------
if __name__ == "__main__":
    # host=0.0.0.0 allows external access if port-forwarded
    app.run(host="0.0.0.0", port=5000, debug=True)
