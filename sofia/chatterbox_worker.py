"""Sofia's voice that can laugh: Chatterbox-Turbo on the GPU, in its own venv and process.

Its torch build clashes with Sofia's .venv, so sofia/tts.py starts this script with the
Chatterbox venv's Python and talks to it over stdin/stdout, one JSON line each way:

    {"text": "[laugh] Oh, sir, stop it!", "path": "/tmp/sofia-x.wav"}  ->  {"ok": true}

Only what speaking needs goes to the GPU, mostly in half precision, so it fits next to the LLM
in ~1.4 GB of VRAM instead of ~3 GB: the S3 tokenizer and speaker encoder (used only to clone a
voice) stay on the CPU, and the text head (used only in training) is dropped.

    python chatterbox_worker.py VOICE.pt                  serve; VOICE.pt is a cloned voice, or "builtin"
    python chatterbox_worker.py --clone REF.wav VOICE.pt  clone the voice in REF.wav (5+ seconds)
"""
import dataclasses
import json
import os
import sys

import soundfile as sf
import torch
from chatterbox.models.s3gen.const import S3GEN_SIL
from chatterbox.tts_turbo import ChatterboxTurboTTS, Conditionals, punc_norm

SPEECH_VOCAB = 6561  # speech tokens at or above this are control tokens, not sound


def _to_gpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to("cuda", dtype=torch.float16) if tensor.is_floating_point() else tensor.to("cuda")


def load(voice: str) -> ChatterboxTurboTTS:
    model = ChatterboxTurboTTS.from_pretrained("cpu")
    if voice != "builtin":
        model.conds = Conditionals.load(voice, map_location="cpu")
    del model.t3.text_head
    model.t3.to("cuda", dtype=torch.float16)
    t3 = model.conds.t3
    for field in dataclasses.fields(t3):
        value = getattr(t3, field.name)
        if torch.is_tensor(value):
            setattr(t3, field.name, _to_gpu(value))
    model.conds.gen = {k: _to_gpu(v) if torch.is_tensor(v) else v for k, v in model.conds.gen.items()}
    model.s3gen.flow.to("cuda", dtype=torch.float16)
    model.s3gen.mel2wav.to("cuda")  # the vocoder stays in full precision
    model.s3gen.trim_fade = model.s3gen.trim_fade.to("cuda")
    return model


@torch.inference_mode()
def speak(model: ChatterboxTurboTTS, text: str):
    """The same steps as ChatterboxTurboTTS.generate, for the split CPU / GPU placement."""
    tokens = model.tokenizer(punc_norm(text), return_tensors="pt").input_ids.to("cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        speech = model.t3.inference_turbo(
            t3_cond=model.conds.t3, text_tokens=tokens, temperature=0.8, top_k=1000, top_p=0.95, repetition_penalty=1.2
        )
        speech = speech[speech < SPEECH_VOCAB]
        speech = torch.cat([speech, torch.tensor([S3GEN_SIL] * 3, device="cuda")])
        mels, _ = model.s3gen.flow.inference(
            token=speech[None], token_len=torch.tensor([len(speech)], device="cuda"),
            finalize=True, noised_mels=None, n_timesteps=2, meanflow=True, **model.conds.gen,
        )
    wav, *_ = model.s3gen.mel2wav.inference(speech_feat=mels.float(), cache_source=torch.zeros(1, 1, 0, device="cuda"))
    wav[:, :len(model.s3gen.trim_fade)] *= model.s3gen.trim_fade
    return model.watermarker.apply_watermark(wav.squeeze(0).cpu().numpy(), sample_rate=model.sr)


def serve(voice: str) -> None:
    replies = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)  # library prints and progress bars go to the log, not into the replies
    model = load(voice)
    speak(model, "Ready.")  # the first call is slow; keep that out of her first reply
    replies.write(json.dumps({"ready": True}) + "\n")
    for line in sys.stdin:
        request = json.loads(line)
        try:
            sf.write(request["path"], speak(model, request["text"]), model.sr)
            replies.write(json.dumps({"ok": True}) + "\n")
        except Exception as error:  # e.g. out of GPU memory: Sofia says this one reply with Kokoro
            torch.cuda.empty_cache()
            replies.write(json.dumps({"error": f"{type(error).__name__}: {error}"}) + "\n")


def clone(reference: str, out: str) -> None:
    model = ChatterboxTurboTTS.from_pretrained("cpu")
    model.prepare_conditionals(reference)
    model.conds.save(out)


if __name__ == "__main__":
    if sys.argv[1] == "--clone":
        clone(sys.argv[2], sys.argv[3])
    else:
        serve(sys.argv[1])
