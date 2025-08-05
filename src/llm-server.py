from fastapi import FastAPI
from typing import List, Dict, Any
import ollama
import logging
from rich.logging import RichHandler

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[RichHandler(markup=True, rich_tracebacks=True)],
)

app = FastAPI()


@app.get("/test")
def test():
    return "FastAPI Server is running"


@app.post("/generate")
def generate(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]] | None = None,
):
    """
    Receives a complete conversation history and tools, gets a response from Ollama,
    and returns the assistant's response.
    """
    response = ollama.chat(
        model="qwen3:latest", messages=messages, tools=tools, think=False
    )

    return {"response": response["message"]}
