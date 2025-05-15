# DocMind RAG System

This is a powerful Streamlit‑based application for uploading, exploring, and learning from your documents through Retrieval‑Augmented Generation (RAG).


## ⭐ Features

* **Document Upload** – Drag and Drop PDFs, DOCX, or TXT files for instant ingestion
* **Question & Answer** – Ask natural‑language questions and receive context aware answers
* **Summarization** – generate concise or detailed overviews of any uploaded document
* **Sentiment Analysis** – gauge the emotional tone across one or many files
* **Custom Analysis** – run topic extraction, keyword detection, or other bespoke tasks
* **Learning‑Style Adaptation** – responses are tailored to Visual, Auditory, Reading/Writing, or Kinesthetic preferences

---

## 🚀 Installation

```bash
# 1 – Install dependencies
pip install -r requirements.txt

# 2 – Configure environment variables
#    Create a .env file in the project root
printf "GROQ_API_KEY=your_groq_api_key" > .env
```
## ▶️ Running the App

```bash
streamlit run app.py
```

## 💡 Usage Guide

| Tab           | What you can do                                                          |
| ------------- | ------------------------------------------------------------------------ |
| **Upload**    | Select one or multiple documents to populate the knowledge base.         |
| **Q\&A**      | Type a question, choose your preferred learning style, then hit **Ask**. |
| **Summarize** | Click **Generate Summary** to receive a high‑level overview.             |
| **Analyze**   | Run topic extraction, sentiment checks, or other custom workflows.       |

### Learning Styles Explained

* **Visual** – diagrams, charts, and spatial layouts
* **Auditory** – spoken‑style explanations and discussions
* **Reading / Writing** – text‑heavy, bullet‑pointed responses
* **Kinesthetic** – step‑by‑step, hands‑on examples and exercises

---

## 📄 🌟 Acknowledgments
- Built with [Streamlit](https://streamlit.io/).
