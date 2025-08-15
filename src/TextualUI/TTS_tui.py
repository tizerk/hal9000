# TTS.py module with some slight tweaks to work better with the TUI
import os
from dotenv import load_dotenv
import pyaudio
import time
import numpy as np
import logging
from pathlib import Path
from typing import Optional
from rich.logging import RichHandler
import threading

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(levelname)s - %(message)s",
    handlers=[RichHandler(markup=True, rich_tracebacks=True)],
)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))


from StyleTTS import msinference
from StyleTTS.app import synthesize


class TTS:
    def __init__(self, character="hal9000"):
        """Text-to-Speech module that takes a text input and outputs audio from StyleTTS inference.

        Args:
            character (str, optional): StyleTTS character voice to be used. Defaults to "hal9000".
        """
        logger.info("Loading StyleTTS reference...")
        self.voice = msinference.compute_style(
            f"{Path(__file__).parent.parent}/StyleTTS/voices/{character}.wav"
        )

        self.p = pyaudio.PyAudio()
        self.stream = None

        self._audio_data = b""
        self._audio_position = 0
        self.stream_lock = threading.Lock()

    def stop_playback(self) -> None:
        """Forcefully stops and closes the current audio stream."""
        with self.stream_lock:
            if self.stream and self.stream.is_active():
                logger.info("Stopping TTS playback.")
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None

    def text_to_speech(
        self, text: str, interrupt_event: Optional[threading.Event] = None
    ) -> None:
        """Runs StyleTTS inference on provided text.

        Args:
            text (str): Message for the TTS module to read
            interrupt_event (threading.Event): Event that interrupts TTS playback
        """
        logger.info("Running TTS Inference...")
        start = time.perf_counter()
        outputs = synthesize(text=text, voice=self.voice, lngsteps=20)
        end = time.perf_counter()
        logger.info(f"TTS inference time: {(end - start):.3f} seconds")

        wav_data = outputs[1]
        wav_data = wav_data / np.max(np.abs(wav_data))
        self._audio_data = (wav_data * 32767).astype(np.int16).tobytes()

        with self.stream_lock:
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=24000,
                output=True,
            )

        logger.info("Playing audio...")
        chunk_size = 1024
        try:
            for i in range(0, len(self._audio_data), chunk_size):
                if interrupt_event and interrupt_event.is_set():
                    logger.info("TTS playback interrupted by event.")
                    break
                with self.stream_lock:
                    if self.stream:
                        try:
                            self.stream.write(self._audio_data[i : i + chunk_size])
                        except (IOError, OSError) as e:
                            logger.warning(f"Stream write error during playback: {e}")
                            break
                    else:
                        break

        finally:
            self.stop_playback()

    def close_stream(self) -> None:
        """Closes Pyaudio stream."""
        self.p.terminate()
