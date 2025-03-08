import os
import time
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from IPython.display import Audio  # type: ignore
from scipy.io.wavfile import write

print("Loading model...")
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

while True:
    print("Inference...")
    t0 = time.time()
    tts = input("Enter text: ")
    outputs = model.inference(
        text=tts,
        language="en",
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        enable_text_splitting=True,
    )

    Audio(data=outputs["wav"], rate=24000)
    output_file_path = f"./outputs/output_audio.wav"
    write(output_file_path, 24000, outputs["wav"])
