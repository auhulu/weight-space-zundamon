"""Generate one unsteered Qwen response per question."""

import argparse
import json
import os
from time import perf_counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT.parent.parent / "cache" / "huggingface"))

import torch
from experiment import CFG, dump, load_model

SYSTEM = "日本語で簡潔に、原則1文で答えてください。"
STYLE = (
    "ずんだもん風の、明るく親しみやすい口調で答えてください。"
    "文末には「〜なのだ」か「〜のだ」を使い、"
    "一人称が必要なときは「ぼく」を使ってください。"
)
MAX_NEW_TOKENS = 128
BATCH_SIZE = 8


@torch.inference_mode()
def main(styled, max_new_tokens):
    started = perf_counter()
    system = SYSTEM + STYLE if styled else SYSTEM
    filename = "qwen_zundamon_prompt_128_rerun.json" if styled else "qwen_baseline_128_rerun.json"
    questions = json.loads((ROOT / "data" / "questions.json").read_text(encoding="utf-8"))
    model, tokenizer = load_model()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    responses = []
    for start in range(0, len(questions), BATCH_SIZE):
        batch = questions[start:start + BATCH_SIZE]
        messages = [
            [{"role": "system", "content": system}, {"role": "user", "content": question["prompt"]}]
            for question in batch
        ]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
            padding=True, return_dict=True, return_tensors="pt",
        ).to(model.device)
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, use_cache=True,
        )
        generated = outputs[:, inputs["input_ids"].shape[1]:].cpu()
        eos = model.generation_config.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos)
        for question, tokens in zip(batch, generated.tolist(), strict=True):
            responses.append({
                **question,
                "answer": tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                "hit_limit": len(tokens) == max_new_tokens and eos_ids.isdisjoint(tokens),
            })
        print(f"{len(responses)}/{len(questions)}", flush=True)

    dump(ROOT / "results" / filename, {
        "model": CFG["model"],
        "revision": CFG["revision"],
        "system": system,
        "enable_thinking": False,
        "decoding": "greedy",
        "max_new_tokens": max_new_tokens,
        "batch_size": BATCH_SIZE,
        "elapsed_seconds": round(perf_counter() - started, 2),
        "responses": responses,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zundamon", action="store_true", help="Add Zundamon style instructions.")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    main(args.zundamon, args.max_new_tokens)
