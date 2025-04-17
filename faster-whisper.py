from faster_whisper import WhisperModel

model_size = "small"

# Run on GPU with FP16
# model = WhisperModel(model_size, device="cuda", compute_type="float16")

# # or run on GPU with INT8
# # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# # or run on CPU with INT8
model = WhisperModel(model_size, device="cpu", compute_type="int8")

    
segments, info = model.transcribe("audio.mp3", beam_size=5, vad_filter=True)
print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
transcribed_text = ""
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    transcribed_text += segment.text
print(transcribed_text)

segments, info = model.transcribe("audio_kr.mp3", beam_size=5, vad_filter=True)
print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
transcribed_text = ""
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    transcribed_text += segment.text
print(transcribed_text)