import pytest
import pyaudio
from unittest.mock import patch, MagicMock
from pynput import keyboard

from src.STT import STT


@pytest.fixture
def mocked_stt_instance():
    """
    Creates an STT instance where all external dependencies (whisper, pyaudio, pynput) are mocked.
    """
    with (
        patch("src.STT.WhisperModel") as mock_whisper,
        patch("src.STT.pyaudio.PyAudio") as mock_pyaudio,
        patch("src.STT.keyboard.Listener") as mock_listener,
    ):

        mock_stream = MagicMock()
        mock_pyaudio_instance = mock_pyaudio.return_value
        mock_pyaudio_instance.open.return_value = mock_stream
        mock_pyaudio_instance.get_default_input_device_info.return_value = {
            "name": "mock_mic",
            "index": 0,
        }

        # Instantiate the STT class, which will use all the mocks above
        stt_instance = STT()

        # Attach mocks to the instance for easy access in tests
        stt_instance.mock_whisper = mock_whisper
        stt_instance.mock_pyaudio_instance = mock_pyaudio_instance
        stt_instance.mock_stream = mock_stream
        stt_instance.mock_listener = mock_listener

        yield stt_instance


def test_initialization(mocked_stt_instance):
    """Tests if the STT class initializes its dependencies correctly."""
    stt = mocked_stt_instance

    stt.mock_whisper.assert_called_once_with(
        model_size_or_path="medium.en", device="cuda", compute_type="float16"
    )

    stt.mock_pyaudio_instance.open.assert_called_once_with(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=1024,
        input_device_index=0,
    )


def test_on_press_spacebar():
    """Tests that pressing the spacebar stops the recording."""
    stt = STT.__new__(STT)
    stt.recording = True

    result = stt._on_press(keyboard.Key.space)

    assert stt.recording is False
    assert result is False


def test_close_stream(mocked_stt_instance):
    """Tests that the stream cleanup function works properly."""
    stt = mocked_stt_instance
    stt.close_stream()

    stt.mock_stream.stop_stream.assert_called_once()
    stt.mock_stream.close.assert_called_once()
    stt.mock_pyaudio_instance.terminate.assert_called_once()
