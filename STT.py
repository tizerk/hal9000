from faster_whisper import WhisperModel
import pyaudio
import wave
import os

print("Loading Whisper model...")
# Run on GPU with FP16
model = WhisperModel(
    model_size_or_path="medium.en", device="cuda", compute_type="float16"
)
# # or run on CPU with INT8
# model = WhisperModel(model_size_or_path="small.en", device="cpu", compute_type="int8")

print("Opening Pyaudio input stream...")
p = pyaudio.PyAudio()

input_device = p.get_default_input_device_info()
print(f"Recording from: {input_device['name']}")

stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    input=True,
    frames_per_buffer=1024,
    input_device_index=input_device["index"],
    start=False
)

def speech_to_text() -> str:
    stream.start_stream()
    transcript = ""
    frames = []
    try:
        while True:
            data = stream.read(1024)
            frames.append(data)
    except KeyboardInterrupt:
        print("\nStopping Input...")
        with wave.open("input_audio.wav", "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(16000)
            wf.writeframes(b"".join(frames))
        segments, _ = model.transcribe("input_audio.wav", beam_size=5, vad_filter=True)
        for segment in segments:
            transcript += segment.text
    finally:
        os.remove("input_audio.wav")
        stream.stop_stream()
        return transcript