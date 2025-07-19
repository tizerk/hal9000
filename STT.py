from faster_whisper import WhisperModel
from pynput import keyboard
import pyaudio
import time
import numpy as np
import logging
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()

import warnings

warnings.filterwarnings(
    "ignore",
    message="dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=0.2 and num_layers=1",
)


class STT:
    def __init__(self, model_size="medium.en", device="cuda", compute_type="float16"):
        self.recording = False
        self.frames = []

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

    def _on_press(self, key):
        if key == keyboard.Key.space:
            console.print(
                "[i]Spacebar[/i] pressed, stopping input...\n", style="bold green"
            )
            self.recording = False
            return False

    def speech_to_text(self) -> str:
        console.print(
            "Now listening... Press [i]Spacebar[/i] to stop recording.\n",
            style="bold green",
        )
        self.recording = True
        self.frames = []
        transcript = ""

        listener = keyboard.Listener(on_press=self._on_press)
        listener.start()

        while self.recording:
            data = self.stream.read(1024)
            self.frames.append(data)

        listener.join()

        audio_data = b"".join(self.frames)
        audio_np = (
            np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        )
        start = time.perf_counter()
        segments, _ = self.model.transcribe(audio_np, beam_size=1, vad_filter=True)
        for segment in segments:
            transcript += segment.text
        end = time.perf_counter()
        logger.info(f"Whisper processing time: {(end - start):.3f} seconds")
        return transcript

    def close_stream(self) -> None:
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
