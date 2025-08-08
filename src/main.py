import os
from dotenv import load_dotenv
import sys
import torch
import requests
import logging
from rich.console import Console
from rich.logging import RichHandler

load_dotenv()
console = Console()

logging.basicConfig(
    format="%(levelname)s - %(message)s",
    handlers=[RichHandler(markup=True, rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))


class MismatchFilter(logging.Filter):
    """Filter for hiding unnecessary warning from phonemizer library"""

    def filter(self, record):
        return "words count mismatch" not in record.getMessage()


phonemizer_logger = logging.getLogger("phonemizer")
phonemizer_logger.addFilter(MismatchFilter())

console.print("Starting HAL9000...\n", style="bold green")
from STT import STT
from TTS import TTS

headers = {"Content-Type": "application/json"}
server_url = "http://127.0.0.1:8001"

if __name__ == "__main__":
    tts_module = TTS(character="hal9000")
    gpu = torch.cuda.is_available()
    if gpu:
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}...")
    else:
        logger.info(f"GPU not found, using CPU...")
    stt_module = STT(
        model_size=f'{"medium.en" if gpu else "small.en"}',
        device=f'{"cuda" if gpu else "cpu"}',
        compute_type=f'{"float16" if gpu else "int8"}',
    )
    try:
        while True:
            user_input = stt_module.speech_to_text()
            console.print(
                f"[bold green]User said:[/bold green]\n\t[i]{user_input}[/i]\n"
            )
            console.print("Asking HAL...\n", style="bold green")
            try:
                llm_response = requests.post(
                    f"{server_url}/query",
                    json={"query": user_input},
                    headers=headers,
                )
                response_text = llm_response.json()["response"]
                console.print(
                    f"[bold green]HAL9000 said:[/bold green]\n\t[i]{response_text}[/i]\n"
                )
                tts_module.text_to_speech(response_text)
            except requests.exceptions.JSONDecodeError:
                logger.error("No response from Ollama. Make sure Ollama is running.")
                sys.exit(1)
            except requests.exceptions.ConnectionError:
                logger.error(
                    "No response from the FastAPI servers.  Make sure they are running."
                )
                sys.exit(1)
    except KeyboardInterrupt:
        console.print("User interrupted, [i]exiting...[/i]", style="bold red")
        sys.exit(0)
    finally:
        tts_module.close_stream()
        stt_module.close_stream()
