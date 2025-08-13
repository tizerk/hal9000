import os
from dotenv import load_dotenv
import pyaudio
import time
import numpy as np
import logging
from pathlib import Path
from typing import Tuple
from rich.logging import RichHandler

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
            f"{Path(__file__).parent}/StyleTTS/voices/{character}.wav"
        )

        self.p = pyaudio.PyAudio()
        self.stream = None

        self._audio_data = b""
        self._audio_position = 0

    def _audio_callback(
        self, in_data, frame_count: int, time_info, status
    ) -> Tuple[bytes, int]:
        """Callback function that the PyAudio thread calls to iteratively get more audio data.

        Args:
            frame_count: The number of frames of audio data the stream needs

        Returns:
            Tuple[bytes, int]: Tuple containing an audio chunk and a flag indicating if there are still chunks left to play.
        """
        bytes_to_read = frame_count * self.p.get_sample_size(pyaudio.paInt16)

        chunk = self._audio_data[
            self._audio_position : self._audio_position + bytes_to_read
        ]

        self._audio_position += bytes_to_read

        if len(chunk) < bytes_to_read:
            padding_size = bytes_to_read - len(chunk)
            chunk += b"\x00" * padding_size
            flag = pyaudio.paComplete
        else:
            flag = pyaudio.paContinue

        return (chunk, flag)

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
        self._audio_data = (wav_data * 32767).astype(np.int16).tobytes()
        self._audio_position = 0

        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=24000,
            output=True,
            stream_callback=self._audio_callback,
        )
        logger.info("Playing audio...")
        try:
            while self.stream.is_active():
                time.sleep(0.05)
        except KeyboardInterrupt:
            logger.info("Playback interrupted by user.")
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None

    def close_stream(self) -> None:
        """Closes Pyaudio stream."""
        self.p.terminate()
