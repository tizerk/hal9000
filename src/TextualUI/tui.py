import os
import sys
import torch
import requests
import logging
import threading
from enum import Enum, auto
from dotenv import load_dotenv

from rich.logging import RichHandler
from rich.console import Console
from rich.text import Text

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.widgets import Footer, Header, RichLog, Static
from textual.worker import Worker, get_current_worker, WorkerState

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from STT_tui import STT
from TTS_tui import TTS

load_dotenv()
console = Console()

logging.basicConfig(
    format="%(levelname)s - %(message)s",
    handlers=[RichHandler(markup=True, rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

SERVER_URL = "http://127.0.0.1:8000"
HEADERS = {"Content-Type": "application/json"}


class AppState(Enum):
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    SPEAKING = auto()


class HAL9000(App[None]):
    CSS_PATH = "hal9000.tcss"
    BINDINGS = [
        Binding("space", "toggle_voice", "Speak", show=True),
        Binding("q", "quit", "Quit", show=True, priority=True),
    ]

    def __init__(self):
        super().__init__()
        self.state = AppState.IDLE
        self.stop_listening_event = threading.Event()
        self.stop_speaking_event = threading.Event()
        self.current_worker: Worker | None = None

        console.print("Starting HAL9000...")
        gpu = torch.cuda.is_available()
        if gpu:
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}...")
        else:
            logger.info(f"GPU not found, using CPU...")
        device = "cuda" if gpu else "cpu"
        compute_type = "float16" if gpu else "int8"
        model_size = "medium.en" if gpu else "small.en"

        self.stt_module = STT(
            model_size=model_size, device=device, compute_type=compute_type
        )
        self.tts_module = TTS(character="hal9000")

    def compose(self) -> ComposeResult:
        yield Header(name="HAL 9000")
        with Container(id="main-container"):
            with Vertical(id="status-panel"):
                yield Static("STATUS", id="status-header")
                yield Static("Press SPACE to speak", id="status-display")
            yield RichLog(id="conversation-log", wrap=True, markup=True)
        yield Footer()

    def update_status(self, text: str, state_class: str = ""):
        status_display = self.query_one("#status-display", Static)
        status_display.update(text)
        status_display.remove_class("listening", "processing", "speaking")
        if state_class:
            status_display.add_class(state_class)

    def add_log(self, user: str, text: str):
        log = self.query_one("#conversation-log", RichLog)
        style = "red" if user == "HAL9000" else "white"
        log.write(Text(f"{user}: ", style=style) + Text(text))

    def _start_listening(self):
        self.state = AppState.LISTENING
        self.update_status("Listening...", state_class="listening")
        self.current_worker = self.run_worker(
            self.interaction_worker, exclusive=True, thread=True
        )

    def action_toggle_voice(self) -> None:
        if self.state == AppState.IDLE:
            self._start_listening()
        elif self.state == AppState.LISTENING:
            self.state = AppState.PROCESSING
            self.update_status("Processing...", state_class="processing")
            self.stop_listening_event.set()
        elif self.state == AppState.SPEAKING:
            logger.info("Interrupting HAL's speech.")
            self.tts_module.stop_playback()
            self.stop_speaking_event.set()
            if self.current_worker:
                self.current_worker.cancel()
            self._start_listening()

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.is_cancelled:
            return
        if event.state == WorkerState.SUCCESS:
            self.state = AppState.IDLE
            self.update_status("Press SPACE to speak")
        elif event.state == WorkerState.ERROR:
            self.state = AppState.IDLE
            self.update_status("An error occurred. Press SPACE to try again.")
            logger.error(f"Worker failed: {event.result}")

    def interaction_worker(self) -> None:
        """The main interaction loop running in a background thread."""
        self.stop_listening_event.clear()
        self.stop_speaking_event.clear()

        transcript = self.stt_module.listen_and_transcribe(self.stop_listening_event)
        worker = get_current_worker()

        if not worker.is_cancelled:
            if not transcript:
                self.state = AppState.IDLE
                return

            self.call_from_thread(self.add_log, "Dave", transcript)
            self.state = AppState.PROCESSING
            self.call_from_thread(
                self.update_status, "Processing...", state_class="processing"
            )

            try:
                response = requests.post(
                    f"{SERVER_URL}/query",
                    json={"query": transcript},
                    headers=HEADERS,
                    timeout=90,
                )
                response.raise_for_status()
                response_text = response.json()["response"]

                if worker.is_cancelled:
                    return

                self.call_from_thread(self.add_log, "HAL9000", response_text)

                self.state = AppState.SPEAKING
                self.call_from_thread(
                    self.update_status, "Speaking...", state_class="speaking"
                )
                self.tts_module.text_to_speech(response_text, self.stop_speaking_event)

            except requests.exceptions.RequestException as e:
                error_message = f"Connection failed: {e}"
                if not worker.is_cancelled:
                    self.call_from_thread(self.add_log, "HAL9000", error_message)
                    self.tts_module.text_to_speech(error_message)
            except Exception as e:
                error_message = f"Internal error: {e}"
                if not worker.is_cancelled:
                    self.call_from_thread(self.add_log, "HAL9000", error_message)
                    self.tts_module.text_to_speech(error_message)

    def on_unmount(self) -> None:
        """Clean up resources on exit."""
        logger.info("Closing streams...")
        self.stt_module.close_stream()
        self.tts_module.close_stream()


if __name__ == "__main__":
    app = HAL9000()
    app.run()
