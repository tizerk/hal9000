# STT.py module with some slight tweaks to work better with the TUI
import os
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from pynput import keyboard
import pyaudio
import time
import numpy as np
import logging
from rich.console import Console
from rich.logging import RichHandler
import threading

load_dotenv()

logging.basicConfig(
    format="%(levelname)s - %(message)s",
    handlers=[RichHandler(markup=True, rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
console = Console()

import warnings

warnings.filterwarnings(
    "ignore",
    message="dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=0.2 and num_layers=1",
)


class STT:
    def __init__(self, model_size="medium.en", device="cuda", compute_type="float16"):
        """Speech-to-Text module that records audio from the default audio device and transcribes it with Faster Whisper.

        Args:
            model_size (str, optional): Whisper model to be used. Larger models are more accurate, but slower. Defaults to "medium.en".
            device (str, optional): Device to be used for transcription. Defaults to "cuda".
            compute_type (str, optional): Compute type to be used for transcription. Float16/32 are more precise, but slower. Defaults to "float16".
        """
        self.recording = False
        self.frames = []
        self._stop_event = threading.Event()

        logger.info("Loading Whisper model...")
        self.model = WhisperModel(
            model_size_or_path=model_size, device=device, compute_type=compute_type
        )

        logger.info("Opening Pyaudio input stream...")
        self.p = pyaudio.PyAudio()

        input_device = self.p.get_default_input_device_info()
        logger.info(f"Recording from: {input_device['name']}")

        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024,
            input_device_index=input_device["index"],
        )

    def listen_and_transcribe(self, stop_event: threading.Event) -> str:
        """Records audio until a stop event is set and then transcribes it."""
        self.frames = []
        self._stop_event = stop_event
        self._stop_event.clear()
        self.recording = True

        while not self._stop_event.is_set():
            data = self.stream.read(1024)
            self.frames.append(data)
        self.recording = False

        if not self.frames:
            return ""

        audio_data = b"".join(self.frames)
        audio_np = (
            np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        )

        if audio_np.size == 0:
            return ""

        start = time.perf_counter()
        segments, _ = self.model.transcribe(audio_np, beam_size=1, vad_filter=True)
        transcript = "".join(segment.text for segment in segments)
        end = time.perf_counter()
        logger.info(f"Whisper processing time: {(end - start):.3f} seconds")
        return transcript

    def close_stream(self) -> None:
        """Closes Pyaudio stream."""
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
