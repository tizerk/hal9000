import pytest
from unittest.mock import patch, AsyncMock, MagicMock

pytestmark = pytest.mark.asyncio

from src.MCP.client import MCPOllamaClient

STDIO_CLIENT_PATH = "src.MCP.client.stdio_client"
CLIENT_SESSION_PATH = "src.MCP.client.ClientSession"


@pytest.fixture
def mocked_mcp_env():
    """Pytest fixture that mocks the mcp library dependencies."""
    with (
        patch(STDIO_CLIENT_PATH) as _,
        patch(CLIENT_SESSION_PATH) as mock_session,
    ):
        mock_session_instance = AsyncMock()
        mock_session_instance.list_tools = AsyncMock(return_value=MagicMock())
        mock_session.return_value = mock_session_instance
        yield mock_session_instance


async def test_initialization():
    """Tests that the client initializes correctly."""
    client = MCPOllamaClient(server_url="http://test.server")
    assert client.http_client.base_url == "http://test.server"
    assert len(client.chat_messages) == 1
    assert client.chat_messages[0]["role"] == "system"
    await client.cleanup()


async def test_cleanup(mocked_mcp_env):
    """Tests that the cleanup fucntion calls the close methods properly."""
    client = MCPOllamaClient()
    client.http_client = AsyncMock()
    client.exit_stack = AsyncMock()

    await client.cleanup()

    client.http_client.aclose.assert_awaited_once()
    client.exit_stack.aclose.assert_awaited_once()
