from fastapi.testclient import TestClient

from src.llm_server import app

client = TestClient(app)


def test_health_check():
    """
    Tests the /test endpoint to ensure the server is running.
    FastAPI should return a 200 OK status code.
    """
    response = client.get("/test")
    assert response.status_code == 200
    assert response.json() == "FastAPI Server is running"


def test_generate_invalid_input():
    """
    Tests the /generate endpoint with an invalid request.
    FastAPI should return a 422 Unprocessable Content status code.
    """
    invalid_payload = {"messages": "Invalid Input"}

    response = client.post("/generate", json=invalid_payload)

    assert response.status_code == 422
