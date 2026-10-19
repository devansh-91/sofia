"""Fine-tune Sofia: a LoRA on Qwen3.5-4B, trained on what training/build.py wrote.

    ./scripts/finetune.sh train      (runs this in the training venv, with Sofia stopped: it needs the GPU)

QLoRA in 4 bits, to fit the 6 GB GPU. The LoRA covers the MLP of every layer and the output of
the full-attention layers only: those map one to one onto the GGUF's tensors, so training/merge.py can fold it straight
into the uncensored model Sofia runs on (HauhauCS's Qwen3.5-4B, a light edit of these same
weights), which is only published as GGUF. The loss counts Sofia's replies only.
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForCausalLM

ROOT = Path(__file__).resolve().parent
DATA = Path.home() / ".local/share/sofia/training/data"  # training/build.py writes it
TRAIN = Path.home() / ".local/share/sofia/train"
BASE = TRAIN / "Qwen3.5-4B"
OUT = TRAIN / "lora"
# Only these: in the GGUF the attention inputs are packed (attn_qkv) and the linear attention's
# output (ssm_out) has its heads reordered (cosine 0.07 against out_proj), so a LoRA on them could
# not be added there as is. The MLP and the full-attention output map one to one (cosine 0.9999).
TARGETS = r".*layers\.\d+\.(mlp\.(gate|up|down)_proj|self_attn\.o_proj)$"
MAX_TOKENS = 2048


def encode(tokenizer, messages: list[dict]) -> list[tuple[list[int], list[int]]]:
    """One sample per reply of Sofia's: the chat up to it, rendered as at run time (the template's
    generation prompt with thinking off), then the reply. Labels are -100 except on the reply.
    The template treats the last reply differently from earlier ones, and at run time she always
    writes the last one, so each reply is trained in that position."""
    samples = []
    for index, message in enumerate(messages):
        # Her answer to the background exchange (sofia/memory.py) is boilerplate, not a reply.
        if message["role"] != "assistant" or messages[index - 1]["content"].startswith("[Where things") \
                or messages[index - 1]["content"].startswith("[Things you already know"):
            continue
        prompt = tokenizer.apply_chat_template(messages[:index], tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        reply_ids = tokenizer(message["content"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
        ids = (prompt_ids + reply_ids)[-MAX_TOKENS:]
        labels = ([-100] * len(prompt_ids) + reply_ids)[-MAX_TOKENS:]
        samples.append((ids, labels))
    return samples


def load(path: Path, tokenizer) -> list[tuple[list[int], list[int]]]:
    return [sample for line in path.read_text().splitlines() if line.strip()
            for sample in encode(tokenizer, json.loads(line)["messages"])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--accumulate", type=int, default=8)
    parser.add_argument("--steps", type=int, default=0, help="stop after this many optimizer steps (a smoke test)")
    arguments = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    train, held = load(DATA / "train.jsonl", tokenizer), load(DATA / "eval.jsonl", tokenizer)
    print(f"{len(train)} replies to train on, {len(held)} held out; longest {max(len(i) for i, _ in train)} tokens", flush=True)

    model = Qwen3_5ForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
    )
    # Not peft's prepare_model_for_kbit_training: it casts the 248k-token embedding table to
    # float32, 2.4 GB on its own.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=arguments.rank, lora_alpha=arguments.rank * 2, lora_dropout=0.05,
                                             target_modules=TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=arguments.lr, weight_decay=0.0)
    total = math.ceil(len(train) * arguments.epochs / arguments.accumulate)
    if arguments.steps:
        total = min(total, arguments.steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / 5) * max(0.05, 1 - step / total))

    def loss_of(ids, labels):
        # The reply is the end of the sample, so only its logits are made: all of them would be
        # 1,400 tokens x 248k vocabulary, 1.4 GB in float32.
        first = next(i for i, label in enumerate(labels) if label != -100)
        logits = model(input_ids=torch.tensor([ids], device="cuda"), logits_to_keep=len(ids) - first + 1).logits[0]
        targets = torch.tensor(ids[first:], device="cuda")
        return torch.nn.functional.cross_entropy(logits[:-1].float(), targets)

    def evaluate() -> float:
        model.eval()
        with torch.no_grad():
            losses = [loss_of(*item).item() for item in held]
        model.train()
        return sum(losses) / len(losses)

    print(f"held-out loss before: {evaluate():.3f}", flush=True)
    model.train()
    step, seen, started, order = 0, 0, time.time(), torch.Generator().manual_seed(7)
    while step < total:
        for index in torch.randperm(len(train), generator=order).tolist():
            loss = loss_of(*train[index]) / arguments.accumulate
            loss.backward()
            seen += 1
            if seen % arguments.accumulate:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            if step % 5 == 0 or step == total:
                print(f"step {step}/{total}  loss {loss.item() * arguments.accumulate:.3f}  "
                      f"{time.time() - started:.0f}s  peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
            if step >= total:
                break
    print(f"held-out loss after: {evaluate():.3f}", flush=True)
    model.save_pretrained(OUT)
    print(f"saved the LoRA to {OUT}", flush=True)


if __name__ == "__main__":
    main()
