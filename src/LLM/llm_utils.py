from typing import List, Dict, Any
import ollama


def generate_llm_response(
    messages: List[Dict[str, Any]], tools: List[Dict[str, Any]] | None = None
) -> Dict[str, Any]:
    """
    Receives a complete conversation history and tools, gets a response from Ollama,
    and returns the assistant's response.
    """
    response = ollama.chat(
        model="qwen3:latest", messages=messages, tools=tools, think=False, keep_alive=-1
    )

    return response["message"]
