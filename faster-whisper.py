from faster_whisper import WhisperModel
import pyaudio
import wave
import os


model_size = "medium"

accumulated_transcription = ""


def record_chunk(p, stream, file_path, chunk_length=1):
    frames = []
    for i in range(0, int(16000 / 1024 * chunk_length)):
        data = stream.read(1024)
        frames.append(data)

    wf = wave.open(file_path, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b"".join(frames))
    wf.close()


def transcribe_chunk(model, file_path):
    segments, _ = model.transcribe(file_path, beam_size=5, vad_filter=True)
    text = ""
    for segment in segments:
        text += segment.text
    return text


def transcription():
    try:
        while True:
            chunk = "./chunk.wav"
            record_chunk(p, stream, chunk)
            transcription = transcribe_chunk(model, chunk)
            print(transcription)
            os.remove(chunk)

            accumulated_transcription += transcription + " "
    except KeyboardInterrupt:
        print("Stopping...")
        with open("transcript.txt", "w") as transcript:
            transcript.write(accumulated_transcription)
    finally:
        print(f"Full Transcript: {accumulated_transcription}")
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    # Run on GPU with FP16
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    # # or run on CPU with INT8
    # model = WhisperModel(model_size, device="cpu", compute_type="int8")

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=1024,
    )
