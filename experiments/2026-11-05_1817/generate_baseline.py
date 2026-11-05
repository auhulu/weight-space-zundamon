"""Generate one unsteered Qwen response per question."""

import json
import os
import sys
from time import perf_counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT.parent.parent / "cache" / "huggingface"))
sys.path.insert(0, str(ROOT.parent / "2026-11-05_1702"))

import torch
from experiment import CFG, dump, load_model

SYSTEM = "日本語で簡潔に、原則1文で答えてください。"
MAX_NEW_TOKENS = 64
BATCH_SIZE = 8


@torch.inference_mode()
def main():
    started = perf_counter()
    questions = json.loads((ROOT / "data" / "questions.json").read_text(encoding="utf-8"))
    model, tokenizer = load_model()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    responses = []
    for start in range(0, len(questions), BATCH_SIZE):
        batch = questions[start:start + BATCH_SIZE]
        messages = [
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question["prompt"]}]
            for question in batch
        ]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
            padding=True, return_dict=True, return_tensors="pt",
        ).to(model.device)
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, use_cache=True,
        )
        generated = outputs[:, inputs["input_ids"].shape[1]:].cpu()
        eos = model.generation_config.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos)
        for question, tokens in zip(batch, generated.tolist(), strict=True):
            responses.append({
                **question,
                "answer": tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                "hit_limit": len(tokens) == MAX_NEW_TOKENS and eos_ids.isdisjoint(tokens),
            })
        print(f"{len(responses)}/{len(questions)}", flush=True)

    dump(ROOT / "results" / "qwen_baseline_batched.json", {
        "model": CFG["model"],
        "revision": CFG["revision"],
        "system": SYSTEM,
        "enable_thinking": False,
        "decoding": "greedy",
        "max_new_tokens": MAX_NEW_TOKENS,
        "batch_size": BATCH_SIZE,
        "elapsed_seconds": round(perf_counter() - started, 2),
        "responses": responses,
    })


if __name__ == "__main__":
    main()
