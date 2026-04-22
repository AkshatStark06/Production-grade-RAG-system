---
title: RAG Production App
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# 🧠 Production-Grade RAG System (₹0 Cost)

A fully local, production-style Retrieval-Augmented Generation (RAG) system built using only open-source tools — no paid APIs.

---

## 🚀 Features

- 🔍 Hybrid Retrieval
  - BM25 (keyword search)
  - Vector Search (semantic search)

- 🎯 Re-ranking
  - Cross-encoder model for better relevance

- 📄 Document Ingestion
  - PDF + Text support
  - Smart chunking

- 🧠 LLM (Fully Local)
  - Hugging Face models (no API)

- ⚙️ Config-driven system
  - YAML-based configuration
  - Prompt versioning support

- 📊 Evaluation
  - RAGAS-based evaluation
  - Golden dataset testing

- 🔁 CI/CD
  - GitHub Actions for automated evaluation

- 💻 UI
  - Streamlit-based chat interface

---

## 🧱 Tech Stack

- LangChain / LlamaIndex
- FAISS / ChromaDB
- Sentence Transformers
- Rank-BM25
- Hugging Face Transformers
- RAGAS
- Streamlit

---

## 💰 Cost Constraint

✅ Built with ₹0 cost  
❌ No OpenAI / paid APIs  
❌ No billing / tokens  

---

## 📁 Project Structure
rag-project/
│
├── data/
├── ingestion/
├── retrieval/
├── reranker/
├── llm/
├── evaluation/
├── config/
├── ui/
├── tests/
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore   


---

## ⚠️ Limitations

- Uses smaller open-source LLMs (not GPT-level)
- Performance depends on local machine
- Evaluation may require tuning for full offline support

---

## 🎯 Goal

To demonstrate a **production-grade RAG architecture** using completely free tools.

---

## 👨‍💻 Author

Akshat Srivastava