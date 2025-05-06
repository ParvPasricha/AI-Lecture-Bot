# AI Lecture Summarizer & Question Generator Bot

**Status:** 🚧 Work in Progress (Core NLP functionality implemented; web search & audio input coming)

A prototype Python/Flask application that:

1. **Summarizes** lecture text using a pre-trained BART model.
2. **Extracts** key topic sentences from the summary.
3. **Generates** exam-style questions for each topic using a T5-based question generator.

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/AI-Lecture-Bot.git
cd AI-Lecture-Bot
```

### 2. Create & activate a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
# source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install torch transformers sentencepiece flask
```

### 4. Run the application
```bash
python app.py
```

### 5. Test the API
Send a POST request to the `/summarize` endpoint:
```bash
curl -X POST http://localhost:5000/summarize \
     -H "Content-Type: application/json" \
     -d '{"text":"Your lecture text here..."}'
```

You should receive a JSON response with `summary`, `key_topics`, and `questions` fields.

---

## 📬 API Reference

| Method | Endpoint     | Request Body                | Response Body                                                   |
|:-------|:-------------|:----------------------------|:---------------------------------------------------------------|
| POST   | `/summarize` | `{ "text": "..." }`     | `{ "summary": str, "key_topics": [str], "questions": [str] }` |

- Returns **400 Bad Request** if the `text` field is missing or empty.

---

## 🔧 How It Works

1. **`summarize_text(text)`**
   - Uses HuggingFace's BART model to condense the input lecture into ~200 tokens.

2. **`break_into_topics(summary)`**
   - Splits the generated summary by sentences, treating each as a separate topic.

3. **`generate_questions(topics)`**
   - For each topic, constructs a T5 prompt to generate a single exam-style question.

4. **Flask App**
   - Hosts a `/summarize` POST endpoint that orchestrates the above steps and returns structured JSON.

---

## 🚧 Next Milestones

- **Web Search Integration**: Enrich questions with live context via SerpAPI or similar.
- **Audio Input**: Transcribe spoken lectures using OpenAI Whisper or Google Speech-to-Text.
- **Visualizations**: Automatically generate charts/graphs for any numeric data in lectures.
- **Frontend UI**: Build a simple HTML/JS or Streamlit interface for non-technical users.
- **Expanded Question Types**: Add MCQs with distractors, coding problems, and long-answer prompts.

Contributions are welcome—feel free to fork, branch, and submit a pull request!

---

## 🤝 Contributing

1. Fork the repository.
2. Create a feature branch to keep work isolated from the main branch:
```bash
git checkout -b feature/your-feature-name
```
3. Make your changes & commit:
```bash
git add .
git commit -m "Describe your feature or fix"
```
4. Push to your fork & open a Pull Request targeting the `main` branch. Your contributions will live in your branch until reviewed and merged into the main project.

We’ll review, discuss feedback, and merge—thank you for helping improve this project!

---

