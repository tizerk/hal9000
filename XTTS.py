import os
import time
import wave
import pyaudio
import torch
import torchaudio
import numpy as np
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

class XTTS:
    # Disable Deepspeed if running torch2.7.0+cu12.8
    def __init__(self, config_path="./XTTS/config.json", deepspeed=False, cuda=True):
        print("Loading XTTS model...")
        config = XttsConfig()
        config.load_json(config_path)
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(config, checkpoint_dir="./XTTS/", use_deepspeed=deepspeed)
        if cuda:
            self.model.cuda()

        print("Computing speaker latents...")
        self.reference_audios = [
            "./XTTS/samples/sample1.wav",
            "./XTTS/samples/sample2.wav",
            "./XTTS/samples/sample3.wav",
        ]
        self.gpt_cond_latent, self.speaker_embedding = self.model.get_conditioning_latents(
            audio_path=self.reference_audios
        )
    
    
    def text_to_speech(self, text) -> None:
        print("Running TTS Inference...")
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
        outputs = self.model.inference(
            text=text,
            language="en",
            gpt_cond_latent=self.gpt_cond_latent,
            speaker_embedding=self.speaker_embedding,
            enable_text_splitting=True,
        )

        wav_data = outputs["wav"]
        wav_data = wav_data / np.max(np.abs(wav_data))
        audio_int16 = (wav_data * 32767).astype(np.int16)
        
        self.stream.write(audio_int16.tobytes())
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()