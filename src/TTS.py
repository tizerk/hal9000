import pyaudio
import time
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

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
            f"{Path(__file__).parent}/StyleTTS/voices/{character}.wav"
        )

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16, channels=1, rate=24000, output=True
        )

    def text_to_speech(self, text: str) -> None:
        """Runs StyleTTS inference on provided text.

        Args:
            text (str): Message for the TTS module to read
        """
        logger.info("Running TTS Inference...")
        start = time.perf_counter()
        outputs = synthesize(text=text, voice=self.voice, lngsteps=20)
        end = time.perf_counter()
        logger.info(f"TTS inference time: {(end - start):.3f} seconds")

        wav_data = outputs[1]
        wav_data = wav_data / np.max(np.abs(wav_data))
        audio_int16 = (wav_data * 32767).astype(np.int16)
        self.stream.write(audio_int16.tobytes())

    def close_stream(self) -> None:
        """Closes Pyaudio stream."""
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
