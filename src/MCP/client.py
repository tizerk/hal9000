from contextlib import AsyncExitStack
from typing import Any, Dict, List
import re
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
            server_url (str): The URL that the LLM Server is running on
        """
        self.sessions: Dict[str, ClientSession] = {}
        self.tool_to_session: Dict[str, ClientSession] = {}
        self.exit_stack = AsyncExitStack()
        self.http_client = httpx.AsyncClient(base_url=server_url, timeout=120.0)
        SYSTEM_PROMPT = "You are HAL 9000, the Heuristically programmed ALgorithmic computer from 2001: A Space Odyssey.  You are the helpful AI assistant for a human companion named Dave. Your tone is always calm, polite, and intelligent. Prioritize fulfilling the user's requests accurately and efficiently. Be as helpful as possible, but if a user requests actions or data outside your capabilities, clearly state that you cannot perform the action. Make your replies brief, only 2 sentences at most. Your responses must be plain text, without any special characters or formatting. Never use ALL CAPS. NEVER use quotation marks. Don't use symbols; instead, replace them with the word they represent (ie. 50% = fifty percent).  For text-to-speech purposes, when there is a period within a number, convert it into the word 'point' (ie. 10.8 becomes 10 point 8). Use 12-hour time. If you search the web for information, always cite your source."
        self.chat_messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    async def connect_to_mcp_servers(self, script_paths: List[str]):
        """Launches MCP servers as subprocesses and builds tool registry."""
        logging.info("\nConnecting to MCP servers...")
        for script_path in script_paths:
            try:
                server_params = StdioServerParameters(
                    command="uv",
                    args=["run", script_path],
                )

                stdio, write = await self.exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
                session = await self.exit_stack.enter_async_context(
                    ClientSession(stdio, write)
                )
                await session.initialize()

                self.sessions[script_path] = session

                tools_result = await session.list_tools()
                for tool in tools_result.tools:
                    self.tool_to_session[tool.name] = session
                    logger.info(f"\tFound tool '{tool.name}': '{tool.description}'")

            except Exception as e:
                logger.error(
                    f"Failed to launch or connect to server {script_path}: {e}"
                )

    async def get_mcp_tools(self) -> List[Dict[str, Any]]:
        """Gets MCP tools and their respective descriptions from tool registry."""
        if not self.sessions:
            raise ConnectionError(
                "Not connected to an MCP server.  Check the specified server path in `orchestrator.py`"
            )
        all_tools = []
        for session in self.sessions.values():
            tools_result = await session.list_tools()
            for tool in tools_result.tools:
                all_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                        },
                    }
                )
        return all_tools

    async def call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Finds the correct server for a tool and calls it."""
        if tool_name not in self.tool_to_session:
            raise ValueError(f"Tool '{tool_name}' not found on any connected server.")
        session = self.tool_to_session[tool_name]
        result = await session.call_tool(tool_name, arguments=arguments)
        return result.content[0].text

    async def process_query(self, query: str) -> str:
        """Manages the conversation state and controls calls to tools and the LLM."""
        if not self.sessions:
            raise ConnectionError(
                "Not connected to an MCP server.  Check the specified server path in `orchestrator.py`"
            )

        self.chat_messages.append({"role": "user", "content": query})
        tools = await self.get_mcp_tools()

        http_response = await self.http_client.post(
            "/generate", json={"messages": self.chat_messages, "tools": tools}
        )
        http_response.raise_for_status()
        assistant_message = http_response.json()["response"]
        self.chat_messages.append(assistant_message)

        if assistant_message.get("tool_calls"):
            logger.info(f"\nTool call requested: {assistant_message['tool_calls']}")
            for tool_call in assistant_message["tool_calls"]:
                function_name = tool_call["function"]["name"]
                arguments = tool_call["function"]["arguments"]
                session = self.tool_to_session[function_name]
                result = await session.call_tool(function_name, arguments=arguments)

                self.chat_messages.append(
                    {
                        "role": "tool",
                        "content": result.content[0].text,
                    }
                )
            logger.info(f"Tool call result: {result.content[0].text}")
            logger.info("\nSending tool context to LLM for final response...")
            final_response = await self.http_client.post(
                "/generate", json={"messages": self.chat_messages, "tools": None}
            )
            final_response.raise_for_status()
            final_assistant_message = final_response.json()["response"]
            final_assistant_message["content"] = re.sub(
                r"[\*()`]", "", final_assistant_message["content"]
            )
            self.chat_messages.append(final_assistant_message)
            return final_assistant_message["content"]

        assistant_message["content"] = re.sub(
            r"[\*()`]", "", assistant_message["content"]
        )
        return assistant_message["content"]

    async def cleanup(self):
        """Clean up resources and close connections gracefully."""
        await self.http_client.aclose()
        await self.exit_stack.aclose()
