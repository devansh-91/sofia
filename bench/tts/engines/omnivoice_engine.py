"""OmniVoice (k2-fsa, the model behind VoiceStudio): zero-shot cloning of the reference voice.

Its inline [laughter] and [sigh] tags stand in for the {laugh} / {sigh} markers.
"""
import sys
from pathlib import Path

import torch
from omnivoice import OmniVoice

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

# Same pacing Sofia gives Piper (sofia/tts.py STYLES).
SPEED = {"sad": 0.92, "lonely": 0.92, "tired": 0.92, "anxious": 0.94, "stressed": 0.94, "angry": 0.94,
         "happy": 1.08, "bored": 1.05}


def load(device: str):
    cuda = device == "cuda"
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0" if cuda else "cpu",
                                      dtype=torch.float16 if cuda else torch.float32)
    ref_wav, ref_text = common.reference()
    common.notes["model"] = "k2-fsa/OmniVoice " + ("fp16" if cuda else "fp32") + ", num_step 32"
    # Without inference_mode, encoding the reference keeps autograd activations: >4 GB for a 10 s clip.
    with torch.inference_mode():
        return model, model.create_voice_clone_prompt(ref_audio=str(ref_wav), ref_text=ref_text)


def synth(loaded, line: dict):
    model, voice = loaded
    text = common.fill(line["devanagari"], laugh="[laughter]", sigh="[sigh]")
    with torch.inference_mode():
        audio = model.generate(text=text, language="hi", voice_clone_prompt=voice,
                               speed=SPEED.get(line["mood"], 1.0))
    return audio[0], 24000


if __name__ == "__main__":
    common.run("omnivoice", load, synth)
