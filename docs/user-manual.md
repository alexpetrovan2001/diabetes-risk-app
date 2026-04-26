# User Manual — Diabetes Risk Prediction System

> Last updated: April 2026

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [First-time Setup](#2-first-time-setup)
3. [Starting the Application](#3-starting-the-application)
4. [Using the Application](#4-using-the-application)
5. [Available npm Commands](#5-available-npm-commands)
6. [API Reference](#6-api-reference)
7. [Running Tests](#7-running-tests)
8. [Re-training the Model](#8-re-training-the-model)
9. [Rebuilding the RAG Index](#9-rebuilding-the-rag-index)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

Install the following tools before anything else.

| Tool | Version | Download |
|---|---|---|
| Python | 3.11 or higher | <https://www.python.org/downloads/> |
| Node.js + npm | 18 or higher | <https://nodejs.org/> |
| Git | any | <https://git-scm.com/> |
| Ollama | latest | <https://ollama.com/download> |

Verify your installs:

```powershell
python --version      # 3.11+
node --version        # v18+
npm --version         # 9+
ollama --version      # any
```

---

## 2. First-time Setup

Run these steps **once** after cloning the repository.

### 2.1 Clone the repository

```powershell
git clone https://github.com/alexpetrovan2001/diabetes-risk-app.git
cd diabetes-risk-app
```

### 2.2 Create and populate the Python virtual environment

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt
cd ..
```

> **Note (Windows execution policy):** If you get a script error, run this first:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
> ```

### 2.3 Install frontend dependencies

```powershell
npm --prefix frontend install
```

### 2.4 Install root-level dev dependencies (concurrently)

```powershell
npm install
```

### 2.5 Pull the local LLM model via Ollama

Make sure Ollama is running (it starts automatically as a background service on Windows after installation), then pull the model:

```powershell
ollama pull llama3.2:1b
```

This downloads ~1.3 GB. Run it once; the model is cached locally.

### 2.6 Build the RAG knowledge index

This only needs to run once (or again if you update the knowledge base):

```powershell
python backend/app/rag/ingest.py
```

Expected output:
```
Loaded 1 source file(s)
Split into N chunks
Building FAISS index...
Saved index to models/faiss_index/
```

### 2.7 Verify the trained model exists

```powershell
Test-Path models/diabetes_model.joblib    # should print True
Test-Path models/faiss_index/index.faiss  # should print True
```

If the model is missing, retrain it:

```powershell
python ml/train_model.py
```

---

## 3. Starting the Application

### Option A — One command (recommended)

From the project root:

```powershell
npm start
```

This starts **both** the backend and frontend concurrently with coloured output:

```
[UI]  VITE v8.x ready on http://localhost:5173
[API] INFO: Uvicorn running on http://127.0.0.1:8000
```

Press `Ctrl+C` to stop both processes at once.

> **Requirement:** Ollama must already be running as a background service (it starts automatically on Windows at login).

---

### Option B — Start services separately

Open two terminal windows.

**Terminal 1 — Backend:**

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

**Terminal 2 — Frontend:**

```powershell
cd frontend
npm run dev
```

---

### Service URLs

| Service | URL |
|---|---|
| Frontend (UI) | <http://localhost:5173> |
| Backend API | <http://127.0.0.1:8000> |
| Interactive API docs | <http://127.0.0.1:8000/docs> |

---

## 4. Using the Application

### 4.1 Making a prediction

1. Open <http://localhost:5173> in your browser.
2. Fill in the **Prediction Input** form with the patient's parameters:

| Field | Description | Typical range |
|---|---|---|
| Pregnancies | Number of pregnancies | 0 – 17 |
| Glucose | Plasma glucose concentration (mg/dL) | 0 – 200 |
| Blood Pressure | Diastolic blood pressure (mmHg) | 0 – 122 |
| Skin Thickness | Triceps skinfold thickness (mm) | 0 – 99 |
| Insulin | 2-hour serum insulin (μU/mL) | 0 – 846 |
| BMI | Body mass index (kg/m²) | 0 – 67 |
| Diabetes Pedigree Function | Family history score | 0.08 – 2.42 |
| Age | Age in years | 21 – 81 |

3. Click **Generate Prediction**.
4. The **Prediction Result** card shows:
   - Risk level badge (High / Low)
   - Probability score
   - Short interpretation message
   - AI-generated explanation (auto-fetched from the RAG module — takes ~15–30 s)

### 4.2 Asking a diabetes question

1. Scroll to the **Ask a Question** panel.
2. Type a question about diabetes (minimum 3 characters, maximum 500).
3. Click **Ask**.
4. The RAG module retrieves relevant knowledge chunks and generates an answer using the local LLM (~15–30 s).

**Example questions:**
- *What does a high glucose level mean for diabetes risk?*
- *How does BMI affect diabetes risk?*
- *What lifestyle changes can reduce diabetes risk?*
- *What is the diabetes pedigree function?*

### 4.3 Viewing prediction history

The **Prediction History** table at the bottom shows all previously saved predictions, ordered newest first. Columns: ID, Glucose, BMI, Age, Risk, Probability, Created At.

---

## 5. Available npm Commands

Run these from the **project root** (`diabetes-risk-app/`):

| Command | What it does |
|---|---|
| `npm start` | Start frontend + backend concurrently |
| `npm run start:frontend` | Start only the React frontend |
| `npm run start:backend` | Start only the FastAPI backend |
| `npm run test:backend` | Run the full pytest test suite |
| `npm run ingest` | Rebuild the FAISS RAG index |
| `npm run train` | Retrain the ML model |

---

## 6. API Reference

All requests/responses are JSON. Base URL: `http://127.0.0.1:8000`

### `GET /health`

```json
// Response 200
{ "status": "ok" }
```

---

### `POST /predict`

**Request body:**

```json
{
  "pregnancies": 2,
  "glucose": 138,
  "blood_pressure": 72,
  "skin_thickness": 35,
  "insulin": 0,
  "bmi": 33.6,
  "diabetes_pedigree_function": 0.627,
  "age": 47
}
```

**Response 200:**

```json
{
  "prediction": 1,
  "risk_label": "high",
  "probability": 0.7842,
  "message": "High risk of diabetes detected. Please consult a healthcare professional."
}
```

---

### `POST /explain`

**Request body** — the prediction result merged with the input values:

```json
{
  "prediction": 1,
  "risk_label": "high",
  "probability": 0.78,
  "pregnancies": 2,
  "glucose": 165,
  "blood_pressure": 80,
  "skin_thickness": 35,
  "insulin": 0,
  "bmi": 34.2,
  "diabetes_pedigree_function": 0.63,
  "age": 45
}
```

**Response 200:**

```json
{
  "explanation": "Based on the input values, the diabetes risk prediction result indicates..."
}
```

---

### `POST /ask`

**Request body:**

```json
{
  "question": "What does a high glucose level mean for diabetes risk?"
}
```

Constraints: `question` must be between 3 and 500 characters.

**Response 200:**

```json
{
  "answer": "A high fasting plasma glucose level of 126 mg/dL or higher..."
}
```

---

### `GET /predictions`

Returns all stored predictions, newest first.

```json
[
  {
    "id": 1,
    "pregnancies": 2,
    "glucose": 138,
    "blood_pressure": 72,
    "skin_thickness": 35,
    "insulin": 0,
    "bmi": 33.6,
    "diabetes_pedigree_function": 0.627,
    "age": 47,
    "prediction": 1,
    "risk_label": "high",
    "probability": 0.7842,
    "message": "...",
    "model_version": "logistic-regression-v1",
    "created_at": "2026-04-26T17:30:00"
  }
]
```

---

## 7. Running Tests

```powershell
npm run test:backend
```

Or directly:

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

**Test coverage:**

| Test | What it covers |
|---|---|
| `test_health_check` | GET /health returns 200 |
| `test_predict_endpoint` | POST /predict returns valid schema |
| `test_explain_endpoint` | POST /explain returns explanation (mocked LLM) |
| `test_ask_endpoint` | POST /ask returns answer (mocked LLM) |
| `test_ask_endpoint_rejects_short_question` | POST /ask rejects question < 3 chars |

---

## 8. Re-training the Model

If you modify the dataset or want to retrain:

```powershell
npm run train
# or
python ml/train_model.py
```

This overwrites `models/diabetes_model.joblib` and `models/feature_columns.joblib`.

To compare models before committing to one:

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
cd ..
python ml/evaluate_models.py       # 5-model comparison table
python ml/preprocess_evaluate.py   # imputation analysis
python ml/feature_importance.py    # feature ranking
```

---

## 9. Rebuilding the RAG Index

If you update `data/rag_sources/diabetes_knowledge.txt`:

```powershell
npm run ingest
# or
python backend/app/rag/ingest.py
```

This re-embeds all chunks and overwrites `models/faiss_index/`.

---

## 10. Troubleshooting

### Backend won't start — `ModuleNotFoundError`

The virtual environment is not activated. Run:

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
```

### `/ask` or `/explain` returns 500 — `FileNotFoundError: FAISS index not found`

The RAG index hasn't been built yet. Run:

```powershell
python backend/app/rag/ingest.py
```

### `/ask` or `/explain` returns 500 — `model requires more system memory`

You're running `llama3.2` (3B) instead of `llama3.2:1b` (1B). Close other applications to free RAM, or verify `rag_service.py` uses `llama3.2:1b`:

```powershell
Select-String "OLLAMA_MODEL" backend/app/rag/rag_service.py
# should print: OLLAMA_MODEL = "llama3.2:1b"
```

### Ollama not responding

Check that the Ollama service is running:

```powershell
ollama list        # should list llama3.2:1b
ollama run llama3.2:1b "hello"   # quick test
```

If not running, start it: open **Ollama** from the Start Menu (system tray icon).

### Frontend can't reach the backend — CORS / network error

Make sure the backend is running on port **8000**. The frontend is hardcoded to `http://127.0.0.1:8000`.

### PowerShell script execution error

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```
