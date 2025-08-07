import pytest
import pyaudio
from unittest.mock import patch, MagicMock

from src.TTS import TTS


@pytest.fixture
def mocked_tts_instance():
    """
    Creates a TTS instance where all external dependencies (styletts, pyaudio) are mocked.
    """
    with (
        patch("src.TTS.msinference.compute_style") as mock_compute_style,
        patch("src.TTS.synthesize") as mock_synthesize,
        patch("src.TTS.pyaudio.PyAudio") as mock_pyaudio,
    ):

        mock_compute_style.return_value = "mock_voice_style"
        mock_pyaudio_instance = mock_pyaudio.return_value

        tts_instance = TTS(character="mock_character")

        tts_instance.mock_synthesize = mock_synthesize
        tts_instance.mock_pyaudio_instance = mock_pyaudio_instance

        yield tts_instance


def test_audio_callback_continue():
    """Tests the audio callback logic when there are chunks left to play."""
    tts = TTS.__new__(TTS)
    tts.p = MagicMock()
    tts.p.get_sample_size.return_value = 2

    tts._audio_data = b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a"
    tts._audio_position = 0

    chunk, flag = tts._audio_callback(None, frame_count=2, time_info=None, status=None)

    assert chunk == b"\x01\x02\x03\x04"
    assert flag == pyaudio.paContinue
    assert tts._audio_position == 4


def test_audio_callback_complete():
    """Tests the audio callback logic when there are no chunks left."""
    tts = TTS.__new__(TTS)
    tts.p = MagicMock()
    tts.p.get_sample_size.return_value = 2

    tts._audio_data = b"\x01\x02\x03\x04\x05\x06"
    tts._audio_position = 4

    chunk, flag = tts._audio_callback(None, frame_count=2, time_info=None, status=None)

    assert chunk == b"\x05\x06\x00\x00"
    assert flag == pyaudio.paComplete
    assert tts._audio_position == 8


def test_close_stream(mocked_tts_instance):
    """Tests that the stream cleanup function works properly."""
    tts = mocked_tts_instance
    tts.close_stream()

    tts.mock_pyaudio_instance.terminate.assert_called_once()
