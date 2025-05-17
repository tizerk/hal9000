import os
import time
import wave
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from IPython.display import Audio
from scipy.io.wavfile import write


print("Loading XTTS model...")
config = XttsConfig()
config.load_json("./XTTS/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="./XTTS/")
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
    print("TTS Inference...")
    outputs = model.inference(
        text=text,
        language="en",
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        enable_text_splitting=True,
    )

    Audio(data=outputs["wav"], rate=24000)
    write("./output_audio.wav", 24000, outputs["wav"])