from pathlib import Path
from fastapi import FastAPI, Body, HTTPException
from typing import Annotated
from contextlib import asynccontextmanager

from .client import MCPOllamaClient

import logging
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[RichHandler(markup=True, rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)

mcp_client = MCPOllamaClient(server_url="http://127.0.0.1:8000")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """On server startup, connect to the MCP tool server.
    On server shutdown, clean up resources."""
    try:
        mcp_servers_scripts = [
            f"{Path(__file__).parent}\\mcp_servers\\mcp_weather\\server.py",
            f"{Path(__file__).parent}\\mcp_servers\\mcp_time\\server.py",
            f"{Path(__file__).parent}\\mcp_servers\\mcp_websearch\\server.py",
        ]
        await mcp_client.connect_to_mcp_servers(mcp_servers_scripts)
    except Exception as e:
        logger.error(f"Could not connect to MCP tool server on startup: {e}")

    yield

    await mcp_client.cleanup()


app = FastAPI(title="MCP Controller", lifespan=lifespan)


@app.post("/query")
async def query(query: Annotated[str, Body(embed=True)]):
    """Receives a query, manages calls to the client, and returns the final response."""
    if not mcp_client.sessions:
        raise HTTPException(status_code=503, detail="MCP Tool Server not connected")
    try:
        final_response = await mcp_client.process_query(query)
        return {"response": final_response}
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))
