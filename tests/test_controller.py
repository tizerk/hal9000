import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

CLIENT_PATH = "src.LLM.controller.mcp_client"


@pytest.fixture
def test_client():
    """
    Pytest fixture that provides a TestClient with a mocked mcp_client.
    """
    with patch(CLIENT_PATH, spec=True) as mock_mcp_client:
        from src.LLM.controller import app

        with TestClient(app) as client:
            yield client, mock_mcp_client


def test_query_success(test_client):
    """Tests the happy path for the /query endpoint.
    FastAPI should return a 200 OK status code.
    """
    client, mock_mcp_client = test_client

    mock_mcp_client.sessions = True
    mock_mcp_client.process_query.return_value = "This is a mock response."

    response = client.post("/query", json={"query": "This is a mock query."})

    assert response.status_code == 200
    assert response.json() == {"response": "This is a mock response."}
    mock_mcp_client.process_query.assert_awaited_once_with("This is a mock query.")


def test_query_server_not_connected(test_client):
    """Tests the case where the MCP tool server is not connected.
    FastAPI should return a 503 Service Unavailable status code.
    """
    client, mock_mcp_client = test_client

    mock_mcp_client.sessions = False

    response = client.post("/query", json={"query": "This will fail."})

    assert response.status_code == 503
    assert response.json() == {"detail": "MCP Tool Server not connected"}


def test_query_processing_error(test_client):
    """Tests the case where the client's process_query method raises an exception.
    FastAPI should return a 500 Internal Server Error status code."""
    client, mock_mcp_client = test_client

    mock_mcp_client.sessions = True
    error_message = "Something went wrong during processing"
    mock_mcp_client.process_query.side_effect = Exception(error_message)

    response = client.post("/query", json={"query": "This will fail."})

    assert response.status_code == 500
    assert response.json() == {"detail": error_message}


def test_lifespan_startup_and_shutdown(test_client):
    """
    Tests that the FastAPI lifespan manager correctly calls the connect function.
    """
    _, mock_mcp_client = test_client

    mock_mcp_client.connect_to_mcp_servers.assert_awaited_once()

    mock_mcp_client.cleanup.assert_not_called()
