# System Architecture

> Diabetes Risk Prediction System — technical design reference  
> Master's Research Project · Babeș-Bolyai University · 2026

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Component Map](#2-component-map)
3. [Machine Learning Module](#3-machine-learning-module)
4. [RAG Module](#4-rag-module)
5. [Backend API](#5-backend-api)
6. [Frontend](#6-frontend)
7. [Database](#7-database)
8. [Request Flows](#8-request-flows)
9. [Technology Decisions](#9-technology-decisions)
10. [Limitations and Known Trade-offs](#10-limitations-and-known-trade-offs)

---

## 1. System Overview

The system is a fully local, offline-capable web application with three independent subsystems:

| Subsystem | Role |
|-----------|------|
| **ML module** | Predicts diabetes risk (binary classification) from 8 clinical inputs |
| **RAG module** | Explains predictions and answers questions using a curated medical knowledge base |
| **Storage module** | Persists every prediction with a timestamp for history and audit |

None of the AI features require an internet connection or a cloud API at runtime. The only external dependency is the Ollama desktop application, which runs the LLM locally.

---

## 2. Component Map

```
┌─────────────────────────────────────────────────────────────┐
│                    Browser  (React + Vite)                   │
│                                                              │
│   ┌──────────────┐   ┌────────────────────┐  ┌───────────┐  │
│   │ Input Form   │   │ Result Card        │  │  Q&A      │  │
│   │ 8 features   │   │ + AI Explanation   │  │  Panel    │  │
│   └──────┬───────┘   └────────┬───────────┘  └─────┬─────┘  │
└──────────┼────────────────────┼──────────────────── ┼───────┘
           │ POST /predict      │ POST /explain        │ POST /ask
┌──────────▼────────────────────▼──────────────────── ▼───────┐
│                    FastAPI  (Uvicorn)                         │
│                                                              │
│  ┌──────────────────┐   ┌──────────────────────────────────┐ │
│  │   ML Service     │   │           RAG Module             │ │
│  │                  │   │                                  │ │
│  │  joblib model ──►│   │  retriever.py                    │ │
│  │  LogisticReg     │   │  ┌──────────────────────────┐   │ │
│  │  → prediction    │   │  │ 1. embed query            │   │ │
│  │  → probability   │   │  │    (all-MiniLM-L6-v2)    │   │ │
│  └──────┬───────────┘   │  │ 2. search FAISS index    │   │ │
│         │               │  │ 3. threshold check       │   │ │
│  ┌──────▼───────────┐   │  │    (L2 dist > 1.2?)      │   │ │
│  │   SQLite DB      │   │  └──────────┬───────────────┘   │ │
│  │   predictions    │   │             │ chunks             │ │
│  └──────────────────┘   │  rag_service.py                  │ │
│                          │  ┌──────────▼───────────────┐   │ │
│                          │  │ 4. build grounded prompt  │   │ │
│                          │  │ 5. call Ollama (llama3.2) │   │ │
│                          │  │ 6. return response text   │   │ │
│                          │  └──────────────────────────┘   │ │
│                          └──────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘

  models/
  ├── diabetes_model.joblib     ← trained LogisticRegression
  ├── feature_columns.joblib    ← ordered feature name list
  └── faiss_index/
      ├── index.faiss           ← FAISS IndexFlatL2 (384 dimensions)
      └── chunks.txt            ← raw text of each indexed chunk

  data/
  ├── diabetes.csv              ← Pima Indians dataset (768 rows)
  └── rag_sources/
      ├── niddk_nih_diabetes.txt
      └── who_diabetes_factsheet.txt
```

---

## 3. Machine Learning Module

### Dataset

The Pima Indians Diabetes Dataset (768 records, 8 features, binary outcome) from the UCI Machine Learning Repository. It contains clinical measurements for female patients of Pima Indian heritage aged 21 and older.

| Feature | Type | Description |
|---------|------|-------------|
| `Pregnancies` | int | Number of pregnancies |
| `Glucose` | int | Plasma glucose concentration (2-hour OGTT, mg/dL) |
| `BloodPressure` | int | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | int | Triceps skin fold thickness (mm) |
| `Insulin` | int | 2-hour serum insulin (µU/mL) |
| `BMI` | float | Body mass index (kg/m²) |
| `DiabetesPedigreeFunction` | float | Family history score (genetic risk proxy) |
| `Age` | int | Age in years |

### Model selection

Five classifiers were evaluated with an 80/20 stratified train/test split (random seed 42):

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| **Logistic Regression** | **0.753** | **0.823** |
| Random Forest | 0.740 | 0.810 |
| Decision Tree | 0.695 | 0.699 |
| K-Nearest Neighbours | 0.714 | 0.775 |
| Support Vector Machine | 0.745 | 0.818 |

Logistic Regression was selected for its best ROC-AUC, full probability output (`predict_proba`), and interpretability — important for a research POC where understanding the model's reasoning matters as much as raw performance.

### Training pipeline (`ml/train_model.py`)

1. Load `data/diabetes.csv` with pandas.
2. Select the 8 feature columns; use `Outcome` as the label.
3. Stratified 80/20 split to preserve class balance (34.9% positive).
4. Fit `LogisticRegression(max_iter=1000)` on the training split.
5. Evaluate on the held-out test split (accuracy + classification report).
6. Serialize model and feature column list with `joblib` to `models/`.

> **Why no zero-value imputation?** The dataset contains physiologically impossible zeros in Glucose, BloodPressure, and BMI, which represent missing values. Testing showed that leaving zeros in place produced a higher ROC-AUC than imputing with column medians. This is a known dataset characteristic documented in related literature.

### Inference (`backend/app/services/prediction_service.py`)

- Models are loaded lazily on first request and cached in module-level globals.
- Input is converted to a pandas DataFrame with columns in the exact order stored in `feature_columns.joblib` — this prevents silent misalignment between training and inference.
- Returns `prediction` (0 or 1), `probability` (float 0–1), `risk_label` ("low"/"high"), and a plain-text `message`.

---

## 4. RAG Module

RAG (Retrieval-Augmented Generation) grounds the LLM's responses in a curated knowledge base instead of relying on the model's training weights. This eliminates hallucinations on specific medical thresholds (e.g. exact A1C values) because the model is only allowed to use the text explicitly provided to it.

The module has three distinct stages:

### Stage 1 — Ingestion (`backend/app/rag/ingest.py`)

Run once to build the vector index from the knowledge base.

```
data/rag_sources/*.txt
        │
        ▼ split on "---" section separator
   text sections
        │
        ▼ sliding word-window (300 words, 50-word overlap)
   chunks[]
        │
        ▼ encode with all-MiniLM-L6-v2 (384-dim float32 vectors)
   embeddings[]
        │
        ▼ add to faiss.IndexFlatL2
        │
        ▼ save
   models/faiss_index/index.faiss   ← the index
   models/faiss_index/chunks.txt    ← raw text, "<<<CHUNK>>>" delimited
```

**Chunking strategy:** The source files use `---` as explicit section boundaries, so the splitter first divides on those, then applies a word-window within each section. The 50-word overlap ensures a phrase that straddles a window boundary still appears complete in at least one chunk.

**Knowledge base sources:** Content is drawn exclusively from official public-domain health authorities:
- **NIDDK/NIH** — What Is Diabetes, Symptoms & Causes, Risk Factors, Tests & Diagnosis, Preventing Type 2 Diabetes (US federal government, public domain)
- **WHO** — Diabetes Fact Sheet, November 2024

The AI-generated placeholder knowledge base was replaced with this official content to ensure factual accuracy on medical thresholds (A1C levels, glucose thresholds, BMI categories, etc.).

### Stage 2 — Retrieval (`backend/app/rag/retriever.py`)

Runs on every `/explain` or `/ask` request.

```
user query (string)
        │
        ▼ encode with all-MiniLM-L6-v2 (same model as ingest)
   query_vector (384-dim float32)
        │
        ▼ faiss.IndexFlatL2.search(query_vector, top_k=4)
   distances[], indices[]
        │
        ▼ best_distance = distances[0][0]
        │
        ├── best_distance > 1.2 → return ([], 1.2+)  ← out-of-scope
        │
        └── best_distance ≤ 1.2 → return top-4 chunk strings
```

**Out-of-scope detection:** `all-MiniLM-L6-v2` produces unit-normalised 384-dimension vectors, so the L2 distance between two vectors is bounded: identical vectors give distance 0, fully orthogonal vectors give distance ~1.41 (≈√2). A threshold of 1.2 was empirically validated:

| Query | Best L2 distance | Decision |
|-------|-----------------|----------|
| "What is a high glucose level?" | 0.94 | in-scope ✓ |
| "Treatment for lung cancer?" | 1.70 | out-of-scope ✗ |
| "How do I cook pasta?" | 2.03 | out-of-scope ✗ |

When out-of-scope, the caller (rag_service) short-circuits and returns a fixed explanatory message — Ollama is never called, saving time and resources.

**Lazy loading:** The FAISS index, chunks list, and embedding model are loaded into module-level globals on the first request and reused for all subsequent requests. This avoids re-loading ~80 MB of model weights on every API call.

### Stage 3 — Generation (`backend/app/rag/rag_service.py`)

```
chunks (list of strings)
        │
        ▼ join with "\n\n---\n\n"
   context block
        │
        ▼ build structured prompt:
        │   "Reference information:\n{context}\n\n
        │    Question: {query}\n\n
        │    Answer using only the reference information above."
        │
        ▼ ollama.chat(model="llama3.2:1b", messages=[system, user])
        │
        ▼ response["message"]["content"]
   answer string
```

**System prompt:** Instructs the model to act as a diabetes education assistant, use only the provided reference text, and always remind the user that the tool is informational — not a medical diagnosis.

**Two entry points:**
- `explain_prediction(...)` — called after `/predict`. Synthesises the input values (glucose, BMI, age, etc.) and the predicted risk into a natural-language query, then retrieves relevant context and generates a 3–5 sentence explanation.
- `answer_question(question)` — called from `/ask`. Passes the user's raw question directly to retrieval and generation.

---

## 5. Backend API

Built with FastAPI, served by Uvicorn. All routes are defined in `backend/app/api/routes.py`.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check — returns `{"status": "ok"}` |
| `POST` | `/predict` | Run ML model, save record to DB, return prediction |
| `POST` | `/explain` | Generate AI explanation for a given prediction result |
| `POST` | `/ask` | Answer a free-text diabetes question via RAG |
| `GET` | `/predictions` | Return all saved prediction records, newest first |

**Why FastAPI?**
- Automatic OpenAPI/Swagger documentation at `/docs` — useful for testing and demonstration.
- Pydantic request validation is built in: invalid field types or out-of-range values (e.g. glucose > 300) are rejected with a structured 422 response before reaching any business logic.
- Async-ready and production-grade, while requiring minimal boilerplate.

**Input validation (`backend/app/schemas/prediction.py`):**

All prediction fields are range-constrained with Pydantic `Field`:

```
pregnancies:               0 – 20
glucose:                   0 – 300 mg/dL
blood_pressure:            0 – 200 mm Hg
skin_thickness:            0 – 100 mm
insulin:                   0 – 1000 µU/mL
bmi:                       0 – 100 kg/m²
diabetes_pedigree_function:0 – 5
age:                       1 – 120
question (AskRequest):     3 – 500 characters
```

---

## 6. Frontend

A single-page React application built with Vite. The UI has three sections that map directly to the three backend subsystems:

| UI Section | Backend endpoint | Trigger |
|------------|-----------------|---------|
| Prediction form | `POST /predict` | Form submit |
| AI Explanation card | `POST /explain` | Automatic after predict |
| Q&A panel | `POST /ask` | User clicks "Ask" |
| Prediction history table | `GET /predictions` | On page load + after predict |

**Why React + Vite?**
- Vite's dev server starts in under one second and has built-in HMR (hot module replacement), making iteration fast.
- React's component model keeps each UI section independent — the prediction form, result card, Q&A panel, and history table each own their own state.
- The frontend has no backend framework concepts leaked into it: it communicates exclusively via HTTP JSON to the FastAPI API.

**API base URL:** Centralised in a single `const API_BASE = "http://127.0.0.1:8000"` constant at the top of `App.jsx`. Changing it for a different deployment requires one edit.

---

## 7. Database

SQLite via SQLAlchemy ORM. A single table `predictions` stores every inference request with its full input vector, prediction result, and timestamp.

**Why SQLite?**
- Zero configuration — no database server process to manage.
- The database file (`backend/diabetes.db`) is created automatically on first startup.
- More than adequate for a local research POC; the table schema is straightforward and can be migrated to PostgreSQL by changing the `DATABASE_URL` string if needed.

**Schema (`backend/app/db/models.py`):**

```
predictions
├── id                        INTEGER PRIMARY KEY
├── pregnancies               INTEGER NOT NULL
├── glucose                   INTEGER NOT NULL
├── blood_pressure            INTEGER NOT NULL
├── skin_thickness            INTEGER NOT NULL
├── insulin                   INTEGER NOT NULL
├── bmi                       FLOAT NOT NULL
├── diabetes_pedigree_function FLOAT NOT NULL
├── age                       INTEGER NOT NULL
├── prediction                INTEGER NOT NULL   (0 or 1)
├── risk_label                VARCHAR NOT NULL   ("low" or "high")
├── probability               FLOAT NOT NULL
├── message                   VARCHAR NOT NULL
├── model_version             VARCHAR NOT NULL   ("logistic-regression-v1")
└── created_at                DATETIME           (server default: now())
```

---

## 8. Request Flows

### Flow A — Prediction + Explanation

```
User submits form
      │
      ▼
POST /predict
  → validate with Pydantic
  → load joblib model (cached)
  → build pandas DataFrame
  → model.predict() + predict_proba()
  → write PredictionRecord to SQLite
  → return PredictionResponse

      │ (frontend auto-calls after success)
      ▼
POST /explain
  → validate with Pydantic
  → build descriptive query string from input values
  → retriever.retrieve(query)
      → encode query → FAISS search → threshold check
  → rag_service builds grounded prompt
  → ollama.chat(llama3.2:1b)
  → return ExplainResponse { explanation: "..." }
```

### Flow B — Q&A

```
User types question → clicks Ask
      │
      ▼
POST /ask
  → validate: 3–500 chars
  → retriever.retrieve(question)
      → encode → FAISS search
      → if best_distance > 1.2: return OUT_OF_SCOPE_MESSAGE immediately
  → rag_service builds prompt with top-4 chunks as context
  → ollama.chat(llama3.2:1b)
  → return AskResponse { answer: "..." }
```

---

## 9. Technology Decisions

### Why `all-MiniLM-L6-v2` for embeddings?

- 384-dimension vectors: compact enough for fast FAISS search, expressive enough for semantic similarity on medical text.
- 6-layer model: very fast inference (~5 ms per query on CPU).
- Strong performance on sentence-level semantic similarity tasks (SBERT benchmarks).
- Runs entirely on CPU — no GPU required.

### Why FAISS (`IndexFlatL2`) instead of a vector database?

- The knowledge base is small (22 chunks). A full vector database (Chroma, Qdrant, Weaviate) would be architectural over-engineering for this scope.
- `IndexFlatL2` is an exact nearest-neighbour search — no approximation error. For 22 vectors this is trivially fast.
- The index is a single `.faiss` file that can be committed to the repository, so a new clone has the index pre-built.

### Why Ollama + `llama3.2:1b`?

- **Fully local:** no API key, no data sent to external services, no cost at runtime.
- **`llama3.2:1b` specifically:** the 1B parameter model requires ~1.3 GB RAM. The 3B parameter `llama3.2` requires ~2.3 GB and causes out-of-memory errors on machines with limited available RAM (observed during development).
- The 1B model is sufficient for this use case because the LLM is only asked to rephrase and summarise text that is explicitly provided in the prompt — it is not asked to reason from parametric knowledge.
- Ollama handles model management, quantisation, and the local HTTP server transparently.

### Why keep the ML model and FAISS index committed to the repo?

- `models/diabetes_model.joblib`, `models/feature_columns.joblib`, and `models/faiss_index/` are checked in.
- This means `npm start` works immediately after setup without requiring a training or ingestion step.
- The training data (`data/diabetes.csv`) is also committed, so reproducibility is guaranteed.

---

## 10. Limitations and Known Trade-offs

| Area | Limitation |
|------|-----------|
| Dataset | 768 records from a single demographic group (Pima Indian women, age ≥ 21). Generalisability to other populations is not established. |
| Zero values | The dataset contains zeros for Glucose, BloodPressure, BMI, and SkinThickness that represent missing data. No imputation is applied because it did not improve ROC-AUC in testing; however, zero-value inputs will still produce a prediction. |
| LLM model size | `llama3.2:1b` is a small model. It follows the grounding prompt reliably but may produce less fluent or nuanced explanations than a larger model. |
| Out-of-scope threshold | The L2 distance threshold of 1.2 was chosen empirically on a small test set. Edge cases (medical questions outside diabetes but semantically close) may be misclassified. |
| No authentication | The API has no authentication layer. This is appropriate for a local research POC but would need to be addressed before any deployment. |
| Windows-only `npm start` | The `start:backend` script uses `.venv\Scripts\uvicorn` (Windows path separators). On macOS/Linux, the path would need to be `.venv/bin/uvicorn`. |
