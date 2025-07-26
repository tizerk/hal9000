from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional
import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import logging
from rich.logging import RichHandler

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[RichHandler(markup=True, rich_tracebacks=True)],
)


class MCPOllamaClient:
    """Client for interacting with the LLM server and MCP tools."""

    def __init__(self, server_url: str = "http://127.0.0.1:8000"):
        """Initialize the Ollama MCP client.

        Args:
            server_url: The URL that the LLM Server is running on
        """
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.http_client = httpx.AsyncClient(base_url=server_url, timeout=120.0)
        self.stdio: Optional[Any] = None
        self.write: Optional[Any] = None

    async def connect_to_server(
        self,
        server_script_path: str = "server.py",
    ):
        """Connect to an MCP server.

        Args:
            server_script_path: Path to the server script to be run with Python.
        """
        server_params = StdioServerParameters(
            command="uv",
            args=["run", server_script_path],
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()

        tools_result = await self.session.list_tools()
        print("\nConnected to MCP server with tools:")
        for tool in tools_result.tools:
            print(f"  - {tool.name}: {tool.description}")

    async def get_mcp_tools(self) -> List[Dict[str, Any]]:
        if not self.session:
            raise ConnectionError(
                "Not connected to an MCP server.  Check the specified server path in `orchestrator.py`"
            )
        tools_result = await self.session.list_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in tools_result.tools
        ]

    async def process_query(self, query: str) -> str:
        """Process a query using an LLM server and available MCP tools."""
        if not self.session:
            raise ConnectionError(
                "Not connected to an MCP server.  Check the specified server path in `orchestrator.py`"
            )

        tools = await self.get_mcp_tools()

        # Call your FastAPI server for the initial response
        http_response = await self.http_client.post(
            "/generate", json={"prompt": query, "tools": tools}
        )
        http_response.raise_for_status()
        response_data = http_response.json()

        assistant_message = response_data["response"]["message"]
        print(assistant_message)

        if assistant_message.get("tool_calls"):
            messages_for_final_call = [
                {"role": "user", "content": query},
                assistant_message,
            ]
            for tool_call in assistant_message["tool_calls"]:
                function_name = tool_call["function"]["name"]
                arguments = tool_call["function"]["arguments"]

                logger.info(f"\nCalling tool: {function_name} with args: {arguments}")

                result = await self.session.call_tool(
                    function_name, arguments=arguments
                )
                tool_output = result.content[0].text
                logger.info(f"Tool response: {tool_output}")

                messages_for_final_call.append({"role": "tool", "content": tool_output})

            final_response = await self.http_client.post(
                "/generate_with_tools",
                json={"messages": messages_for_final_call},
            )
            return final_response.json()["response"]["message"]["content"]

        # If no tool calls, return the original content
        return assistant_message["content"]

    async def cleanup(self):
        """Clean up resources and close connections gracefully."""
        await self.http_client.aclose()
        await self.exit_stack.aclose()
