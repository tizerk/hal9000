import os
import time
import wave
import pyaudio
import torch
import torchaudio
import numpy as np
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from IPython.display import Audio
from scipy.io.wavfile import write


print("Loading XTTS model...")
config = XttsConfig()
config.load_json("./XTTS/config.json")
model = Xtts.init_from_config(config)
# Disable Deepspeed if running torch2.7.0+cu12.8
model.load_checkpoint(config, checkpoint_dir="./XTTS/", use_deepspeed=False)
# Remove the following line if running on CPU
model.cuda()

print("Computing speaker latents...")
reference_audios = [
    "./XTTS/samples/sample1.wav",
    "./XTTS/samples/sample2.wav",
    "./XTTS/samples/sample3.wav",
]
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
    audio_path=reference_audios
)
    
    
def text_to_speech(text) -> None:
    print("Running TTS Inference...")
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
    outputs = model.inference(
        text=text,
        language="en",
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        enable_text_splitting=True,
    )

    wav_data = outputs["wav"]
    wav_data = wav_data / np.max(np.abs(wav_data))
    audio_int16 = (wav_data * 32767).astype(np.int16)
    
    stream.write(audio_int16.tobytes())
    stream.stop_stream()
    stream.close()
    p.terminate()