"""Basic API tests for the redesigned schema. Run with: pytest tests/ -v"""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "age" in data["features"]


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_churn_yes_like_sample():
    """Mirrors first sample row (high support calls, short tenure, Month-to-month, Fiber)."""
    payload = {
        "age": 56,
        "tenure": 1,
        "monthly_charges": 35.17,
        "total_charges": 35.17,
        "contract": "Month-to-month",
        "internet_service": "Fiber Optic",
        "support_calls": 6,
        "payment_method": "Electronic Check",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert data["churn_prediction"] in ("Yes", "No")
    assert 0.0 <= data["churn_probability"] <= 1.0


def test_predict_churn_no_like_sample():
    """Mirrors second sample row (long tenure, One Year contract)."""
    payload = {
        "age": 69,
        "tenure": 58,
        "monthly_charges": 51.17,
        "total_charges": 3008.73,
        "contract": "One Year",
        "internet_service": "Fiber Optic",
        "support_calls": 6,
        "payment_method": "Mailed Check",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["churn_prediction"] in ("Yes", "No")


def test_predict_missing_required():
    payload = {"age": 40, "tenure": 10}  # missing many required fields
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
