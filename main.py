import sys
import requests
import logging
from rich.console import Console
from rich.logging import RichHandler

console = Console()

logging.basicConfig(
    level=logging.ERROR,
    format="%(levelname)s - %(message)s",
    handlers=[RichHandler(markup=True, rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)
console.print("Starting HAL9000...\n", style="bold green")
from STT import STT
from TTS import TTS

headers = {"Content-Type": "application/json"}
server_url = "http://127.0.0.1:8000"

if __name__ == "__main__":

    tts_module = TTS()
    stt_module = STT()
    try:
        while True:
            user_input = stt_module.speech_to_text()
            console.print(
                f"[bold green]User said:[/bold green]\n\t[i]{user_input}[/i]\n"
            )
            console.print("Asking HAL...\n", style="bold green")
            try:
                llm_response = requests.post(
                    f"{server_url}/generate?prompt={user_input}", headers=headers
                )
                tts_module.text_to_speech(llm_response.json()["response"])
                console.print(
                    f"[bold green]HAL9000 said:[/bold green]\n\t[i]{llm_response.json()["response"]}[/i]\n"
                )
            except requests.exceptions.JSONDecodeError:
                logger.error("No response from Ollama. Make sure Ollama is running.")
                sys.exit(1)
            except requests.exceptions.ConnectionError:
                logger.error(
                    "No response from the FastAPI server.  Make sure it's running with `uv run fastapi run llm-server.py`."
                )
                sys.exit(1)
    except KeyboardInterrupt:
        console.print("User interrupted, [i]exiting...[/i]", style="bold red")
        sys.exit(0)
    finally:
        tts_module.close_stream()
        stt_module.close_stream()
