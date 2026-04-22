---
title: RAG Production App
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# 🧠 Production-Grade RAG System (₹0 Cost)

A fully local, production-style Retrieval-Augmented Generation (RAG) system built using open-source tools — **no paid APIs, no external LLM services**.

---

## 🚀 Live Demo

👉 https://huggingface.co/spaces/AkshatSri/rag-production-app

---

## 📌 Project Versions

### 🔹 v2-hf-production (Current - Recommended)
- Deployed on Hugging Face Spaces
- Uses `flan-t5-base` (optimized for free-tier deployment)
- Fully integrated system with UI + backend

**Key Features:**
- Hybrid Retrieval (BM25 + FAISS)
- Cross-encoder reranking
- Streaming responses
- Confidence scoring (normalized)
- Debug mode (context + system transparency)

---

### 🔹 Previous Version
- `step 14.1-caching`
- Includes earlier pipeline stages and docker-compose setup

---

### 🔹 Future Work
- `rag-local-mistral` branch
  - Local LLM (Mistral)
  - Higher performance (GPU / Colab)
  - Advanced experimentation

---

## 🧠 System Overview

This system follows a **production-grade RAG pipeline**:

1. User Query  
2. Query Splitting  
3. Hybrid Retrieval (BM25 + Vector Search)  
4. Cross-Encoder Re-ranking  
5. Context Selection  
6. LLM Generation  
7. Confidence Scoring  

---

## 🏗️ Architecture

![RAG Architecture](architecture.png)

---

## 🚀 Key Features

### 🔍 Hybrid Retrieval
- BM25 → keyword matching
- FAISS → semantic similarity
- Combines both for better recall

---

### 🎯 Re-ranking
- Cross-encoder (`ms-marco-MiniLM-L-6-v2`)
- Improves precision of retrieved documents

---

### 🧠 LLM (Fully Open-Source)
- `google/flan-t5-base`
- Optimized for Hugging Face free-tier deployment

---

### ⚡ Streaming Responses
- Real-time answer generation
- Token-by-token display in UI

---

### 📊 Confidence Scoring (Explainability)
- Derived from reranker scores
- Normalized using sigmoid
- Indicates relevance of retrieved context

---

### 🧪 Evaluation System
- RAGAS-based evaluation
- Golden dataset validation
- Threshold-based filtering

---

### ⚙️ Config-Driven Design
- YAML-based configs
- Prompt versioning
- Easily extensible

---

### 💻 UI (Streamlit)
- Chat interface
- Streaming + Debug mode
- Context visualization
- Confidence display

---

## 🧱 Tech Stack

- FastAPI (Backend)
- Streamlit (Frontend)
- FAISS (Vector DB)
- Sentence Transformers (Embeddings)
- Rank-BM25 (Keyword Retrieval)
- Hugging Face Transformers (LLM)
- RAGAS (Evaluation)
- Docker (Deployment)

---

## 💰 Cost Constraint

✅ ₹0 cost system  
❌ No OpenAI / paid APIs  
❌ No token billing  

---

## 📁 Project Structure
The project follows a modular, production-grade architecture:
rag-project/
│
├── api/ # FastAPI routes & schemas
├── services/ # RAG service layer
│
├── data/ # Source documents (sample.txt)
├── ingestion/ # Document loading & chunking
├── retrieval/ # BM25 + FAISS + hybrid retrieval
├── reranker/ # Cross-encoder reranking
├── llm/ # LLM generation logic
│
├── evaluation/ # RAGAS evaluation pipeline
├── config/ # YAML configs (settings, prompts)
├── utils/ # Logging & helper utilities
│
├── ui/ # Streamlit frontend
│
├── main.py # Pipeline builder
├── pipeline.py # Core RAG pipeline
├── requirements.txt # Dependencies
├── Dockerfile # Hugging Face deployment
└── README.md

---

## ⚠️ Limitations

- Uses smaller open-source LLMs (not GPT-level)
- Limited by free-tier compute (Hugging Face)
- Retrieval quality depends on dataset size

---

## 🎯 Project Goal

To design and deploy a **real-world, production-style RAG system** using only free and open-source tools while maintaining:

- scalability
- interpretability
- modular architecture

---

## 👨‍💻 Author

**Akshat Srivastava**
