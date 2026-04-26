import { useEffect, useState } from "react";
import axios from "axios";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";

const initialForm = {
  pregnancies: 0,
  glucose: 120,
  blood_pressure: 70,
  skin_thickness: 20,
  insulin: 79,
  bmi: 25.6,
  diabetes_pedigree_function: 0.45,
  age: 33,
};

const fieldLabels = {
  pregnancies: "Pregnancies",
  glucose: "Glucose",
  blood_pressure: "Blood Pressure",
  skin_thickness: "Skin Thickness",
  insulin: "Insulin",
  bmi: "BMI",
  diabetes_pedigree_function: "Diabetes Pedigree Function",
  age: "Age",
};

function App() {
  const [formData, setFormData] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [explanation, setExplanation] = useState(null);
  const [explanationLoading, setExplanationLoading] = useState(false);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState(null);
  const [answerLoading, setAnswerLoading] = useState(false);

  const fetchHistory = async () => {
    try {
      const response = await axios.get(`${API_BASE}/predictions`);
      setHistory(response.data);
    } catch (error) {
      console.error("Failed to fetch history:", error);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const handleChange = (event) => {
    const { name, value } = event.target;

    setFormData((prev) => ({
      ...prev,
      [name]: value === "" ? "" : Number(value),
    }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setResult(null);

    try {
      const response = await axios.post(
        `${API_BASE}/predict`,
        formData,
      );

      setResult(response.data);
      await fetchHistory();
      fetchExplanation(response.data, formData);
    } catch (error) {
      console.error("Prediction request failed:", error);
      alert("Prediction request failed. Please verify the input values.");
    } finally {
      setLoading(false);
    }
  };

  const fetchExplanation = async (predResult, inputs) => {
    setExplanationLoading(true);
    setExplanation(null);
    try {
      const response = await axios.post(`${API_BASE}/explain`, {
        prediction: predResult.prediction,
        risk_label: predResult.risk_label,
        probability: predResult.probability,
        ...inputs,
      });
      setExplanation(response.data.explanation);
    } catch (error) {
      console.error("Explanation request failed:", error);
    } finally {
      setExplanationLoading(false);
    }
  };

  const handleAsk = async (event) => {
    event.preventDefault();
    if (!question.trim()) return;
    setAnswerLoading(true);
    setAnswer(null);
    try {
      const response = await axios.post(`${API_BASE}/ask`, {
        question: question.trim(),
      });
      setAnswer(response.data.answer);
    } catch (error) {
      console.error("Ask request failed:", error);
    } finally {
      setAnswerLoading(false);
    }
  };

  const formatProbability = (value) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  const formatDate = (value) => {
    if (!value) {
      return "-";
    }

    return new Date(value).toLocaleString();
  };

  const resultClass =
    result?.risk_label === "high" ? "risk-badge high" : "risk-badge low";

  return (
    <div className="app-shell">
      <div className="app-container">
        <header className="hero-section">
          <div>
            <p className="eyebrow">Research Prototype</p>
            <h1>Diabetes Risk Prediction System</h1>
            <p className="hero-text">
              Educational proof of concept for estimating diabetes risk using
              machine learning and storing prediction results in a database.
            </p>
          </div>
        </header>

        <main className="content-grid">
          <section className="card form-card">
            <div className="card-header">
              <h2>Prediction Input</h2>
              <p>Enter the patient parameters to generate a prediction.</p>
            </div>

            <form onSubmit={handleSubmit} className="prediction-form">
              {Object.keys(initialForm).map((field) => (
                <div key={field} className="form-field">
                  <label htmlFor={field}>{fieldLabels[field]}</label>
                  <input
                    id={field}
                    type="number"
                    step="any"
                    name={field}
                    value={formData[field]}
                    onChange={handleChange}
                    required
                  />
                </div>
              ))}

              <button type="submit" className="primary-button" disabled={loading}>
                {loading ? "Generating..." : "Generate Prediction"}
              </button>
            </form>
          </section>

          <section className="card result-card">
            <div className="card-header">
              <h2>Prediction Result</h2>
              <p>The latest generated result will appear below.</p>
            </div>

            {result ? (
              <div className="result-content">
                <div className={resultClass}>
                  <span className="badge-label">Risk Level</span>
                  <strong>
                    {result.risk_label === "high" ? "High Risk" : "Low Risk"}
                  </strong>
                </div>

                <div className="result-stats">
                  <div className="stat-box">
                    <span>Prediction Value</span>
                    <strong>{result.prediction}</strong>
                  </div>
                  <div className="stat-box">
                    <span>Probability</span>
                    <strong>{formatProbability(result.probability)}</strong>
                  </div>
                </div>

                <div className="message-box">
                  <span>Interpretation</span>
                  <p>{result.message}</p>
                </div>

                {explanationLoading && (
                  <div className="explanation-loading">
                    <div className="spinner" />
                    <span>Generating AI explanation…</span>
                  </div>
                )}

                {explanation && (
                  <div className="explanation-box">
                    <span>AI Explanation</span>
                    <p>{explanation}</p>
                  </div>
                )}
              </div>
            ) : (
              <div className="empty-state">
                No prediction generated yet. Submit the form to see a result.
              </div>
            )}
          </section>
        </main>

        <section className="card qa-card">
          <div className="card-header">
            <h2>Ask a Question</h2>
            <p>Ask anything about diabetes risk factors, symptoms, or prevention.</p>
          </div>

          <form onSubmit={handleAsk} className="qa-form">
            <textarea
              className="qa-input"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="e.g. What does a high glucose level mean for diabetes risk?"
              rows={3}
              maxLength={500}
            />
            <button
              type="submit"
              className="primary-button qa-button"
              disabled={answerLoading || !question.trim()}
            >
              {answerLoading ? "Thinking…" : "Ask"}
            </button>
          </form>

          {answerLoading && (
            <div className="explanation-loading">
              <div className="spinner" />
              <span>Retrieving answer…</span>
            </div>
          )}

          {answer && (
            <div className="explanation-box">
              <span>Answer</span>
              <p>{answer}</p>
            </div>
          )}
        </section>

        <section className="card history-card">
          <div className="card-header">
            <h2>Prediction History</h2>
            <p>Previously saved prediction records from the local database.</p>
          </div>

          {history.length === 0 ? (
            <div className="empty-state">No saved predictions yet.</div>
          ) : (
            <div className="table-wrapper">
              <table>
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Glucose</th>
                    <th>BMI</th>
                    <th>Age</th>
                    <th>Risk</th>
                    <th>Probability</th>
                    <th>Created At</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((item) => (
                    <tr key={item.id}>
                      <td>{item.id}</td>
                      <td>{item.glucose}</td>
                      <td>{item.bmi}</td>
                      <td>{item.age}</td>
                      <td>
                        <span
                          className={
                            item.risk_label === "high"
                              ? "table-badge high"
                              : "table-badge low"
                          }
                        >
                          {item.risk_label === "high" ? "High Risk" : "Low Risk"}
                        </span>
                      </td>
                      <td>{formatProbability(item.probability)}</td>
                      <td>{formatDate(item.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

export default App;