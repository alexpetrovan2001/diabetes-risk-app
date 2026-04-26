# Diabetes Risk Prediction System

> **Educational and research proof-of-concept** — not a medical device.  
> Master's Research Project · Babeș-Bolyai University · 2026  
> Student: Alex Petrovan · Supervisor: prof. univ. dr. Andreica Anca-Mirela  
> GitHub: <https://github.com/alexpetrovan2001/diabetes-risk-app>

---

## What is this project?

An intelligent end-to-end web application that:

1. **Predicts diabetes risk** from 8 clinical parameters using a trained Logistic Regression model (ROC-AUC 0.823 on the Pima Indians Diabetes Dataset).
2. **Stores every prediction** with a timestamp in a local SQLite database and shows history in the UI.
3. **Explains the result** in plain language using a local Retrieval-Augmented Generation (RAG) pipeline — no cloud services required.
4. **Answers diabetes questions** through a Q&A panel powered by the same RAG module.

---

## Research Scope

| Aspect | Detail |
|---|---|
| Dataset | Pima Indians Diabetes Dataset (768 records, 8 features) |
| Problem type | Binary classification (diabetic / not diabetic) |
| Best model | Logistic Regression — ROC-AUC **0.823**, Accuracy **0.753** |
| Models compared | LR, Random Forest, Decision Tree, KNN, SVM |
| RAG embedding model | `all-MiniLM-L6-v2` (sentence-transformers) |
| Vector store | FAISS (faiss-cpu) |
| Local LLM | `llama3.2:1b` via Ollama |
| Scope | Local POC — educational / research only |

**Key research findings:**
- Logistic Regression without zero-value imputation achieved the best ROC-AUC.
- Glucose is the most predictive feature in both LR and Random Forest.
- A fully local RAG pipeline (FAISS + llama3.2:1b) eliminates hallucinations on threshold values by grounding answers in a curated medical knowledge base.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Browser (React + Vite)               │
│  ┌──────────────┐  ┌────────────────┐  ┌─────────────┐  │
│  │ Input Form   │  │ Result + AI    │  │  Q&A Panel  │  │
│  │ (8 features) │  │ Explanation    │  │  /ask       │  │
│  └──────┬───────┘  └───────┬────────┘  └──────┬──────┘  │
└─────────┼──────────────────┼─────────────────┼──────────┘
          │ POST /predict     │ POST /explain    │ POST /ask
┌─────────▼──────────────────▼─────────────────▼──────────┐
│                   FastAPI Backend                         │
│  ┌──────────────┐  ┌────────────────────────────────┐    │
│  │ ML Service   │  │         RAG Module             │    │
│  │ (LR model)   │  │  FAISS ──► Prompt ──► Ollama   │    │
│  └──────┬───────┘  └────────────────────────────────┘    │
│         │                                                 │
│  ┌──────▼───────┐                                        │
│  │  SQLite DB   │  (prediction history)                  │
│  └──────────────┘                                        │
└──────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Version |
|---|---|---|
| Frontend | React + Vite | React 19, Vite 8 |
| HTTP client | Axios | 1.x |
| Backend | FastAPI + Uvicorn | 0.135 / 0.42 |
| ORM | SQLAlchemy | 2.0 |
| Database | SQLite | — |
| ML | scikit-learn | 1.8 |
| Embeddings | sentence-transformers | 5.4 |
| Vector store | faiss-cpu | 1.13 |
| Local LLM | Ollama + llama3.2:1b | — |
| Language | Python 3.14 / Node.js 24 | — |

---

## Quick Start

### Prerequisites
- Python 3.11+ with `venv`
- Node.js 18+
- [Ollama](https://ollama.com) installed and running
- `llama3.2:1b` model pulled: `ollama pull llama3.2:1b`

### One-command start (after setup)

```bash
npm start
```

This starts both the backend (port 8000) and the frontend (port 5173) concurrently.

See [docs/user-manual.md](docs/user-manual.md) for the full setup guide.  
See [docs/architecture.md](docs/architecture.md) for a detailed technical design reference.

---

## Project Structure

```
diabetes-risk-app/
├── backend/
│   ├── app/
│   │   ├── api/routes.py          # All API endpoints
│   │   ├── db/                    # SQLAlchemy models & session
│   │   ├── rag/                   # RAG module (ingest, retriever, service)
│   │   ├── schemas/               # Pydantic request/response models
│   │   └── services/              # ML prediction service
│   └── tests/                     # pytest test suite
├── data/
│   ├── diabetes.csv               # Pima Indians dataset
│   └── rag_sources/               # Medical knowledge base (plain text)
├── docs/                          # Project documentation
├── frontend/src/                  # React application
├── ml/                            # Training & evaluation scripts
└── models/
    ├── diabetes_model.joblib      # Trained LR model
    ├── feature_columns.joblib     # Feature names
    └── faiss_index/               # FAISS vector index + chunks
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/predictions` | List all stored predictions |
| `POST` | `/predict` | Run ML prediction |
| `POST` | `/explain` | Generate AI explanation for a result |
| `POST` | `/ask` | Answer a diabetes question via RAG |

Interactive API docs: <http://127.0.0.1:8000/docs>

---

## Disclaimer

This project is for **educational and research purposes only**.  
It does **not** constitute medical advice or diagnosis and must not replace professional medical consultation.