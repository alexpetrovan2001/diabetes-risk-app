from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint():
    payload = {
        "pregnancies": 2,
        "glucose": 138,
        "blood_pressure": 72,
        "skin_thickness": 35,
        "insulin": 0,
        "bmi": 33.6,
        "diabetes_pedigree_function": 0.627,
        "age": 47,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "risk_label" in data
    assert "probability" in data
    assert "message" in data


def test_explain_endpoint():
    payload = {
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
        "age": 45,
    }

    with patch(
        "app.api.routes.explain_prediction",
        return_value="This result indicates elevated diabetes risk.",
    ):
        response = client.post("/explain", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "explanation" in data
    assert data["explanation"] == "This result indicates elevated diabetes risk."


def test_ask_endpoint():
    with patch(
        "app.api.routes.answer_question",
        return_value="High glucose indicates elevated blood sugar and increased diabetes risk.",
    ):
        response = client.post(
            "/ask",
            json={"question": "What does high glucose mean?"},
        )

    assert response.status_code == 200

    data = response.json()

    assert "answer" in data
    assert data["answer"] == "High glucose indicates elevated blood sugar and increased diabetes risk."


def test_ask_endpoint_rejects_short_question():
    response = client.post("/ask", json={"question": "Hi"})

    assert response.status_code == 422