import os
from dotenv import load_dotenv
from fastapi import FastAPI
from typing import List, Dict, Any
import ollama
import logging
from rich.logging import RichHandler

load_dotenv()
logging.basicConfig(
    format="%(levelname)s - %(message)s",
    handlers=[RichHandler(markup=True, rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

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
        model="qwen3:latest", messages=messages, tools=tools, think=False, keep_alive=-1
    )

    return {"response": response["message"]}
