from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Russian Sentiment Analysis API"
    }


def test_predict():
    response = client.post(
        "/predict",
        json={"text": "Мне очень понравился этот фильм!"},
    )

    assert response.status_code == 200

    data = response.json()

    assert "label" in data
    assert "score" in data

    assert data["label"] in ["POSITIVE", "NEGATIVE"]
    assert 0 <= data["score"] <= 1


def test_predict_negative():
    response = client.post(
        "/predict",
        json={"text": "Это был ужасный фильм, мне совершенно не понравилось."},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["label"] == "NEGATIVE"
    assert 0 <= data["score"] <= 1


def test_predict_without_text():
    response = client.post(
        "/predict",
        json={},
    )

    assert response.status_code == 422