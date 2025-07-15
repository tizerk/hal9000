from faster_whisper import WhisperModel
from pynput import keyboard
import pyaudio
import time
import wave
import os
import logging

logger = logging.getLogger(__name__)

import warnings

warnings.filterwarnings(
    "ignore",
    message="dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=0.2 and num_layers=1",
)


class STT:
    # or run on CPU with INT8 (self, model_size="small.en", device="cpu", compute_type="int8")
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
            start=False,
        )
        print("Listening...\n")

    def _on_press(self, key):
        if key == keyboard.Key.esc:
            print("Esc pressed, stopping input...\n")
            self.recording = False
            return False

    def speech_to_text(self) -> str:
        self.recording = True
        self.frames = []
        transcript = ""
        self.stream.start_stream()

        listener = keyboard.Listener(on_press=self._on_press)
        listener.start()

        while self.recording:
            data = self.stream.read(1024)
            self.frames.append(data)

        listener.join()

        with wave.open("input_audio.wav", "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(16000)
            wf.writeframes(b"".join(self.frames))
        start = time.perf_counter()
        segments, _ = self.model.transcribe(
            "input_audio.wav", beam_size=5, vad_filter=True
        )
        end = time.perf_counter()
        for segment in segments:
            transcript += segment.text
        os.remove("input_audio.wav")
        logger.info(f"Whisper processing time: {(end - start):.3f} seconds")
        self.stream.stop_stream()
        self.stream.close()
        return transcript

    def close_stream(self) -> None:
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
