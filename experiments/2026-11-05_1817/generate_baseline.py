"""Generate one unsteered Qwen response per question."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT.parent.parent / "cache" / "huggingface"))
sys.path.insert(0, str(ROOT.parent / "2026-11-05_1702"))

from experiment import CFG, dump, generate, load_model


def main():
    questions = json.loads((ROOT / "data" / "questions.json").read_text(encoding="utf-8"))
    model, tokenizer = load_model()
    seed = CFG["selection_seed"]
    responses = []
    for question in questions:
        result = generate(model, tokenizer, question["prompt"], seed, greedy=True)
        responses.append({
            **question,
            "answer": result["text"].strip(),
            "hit_limit": result["hit_limit"],
        })
        print(question["id"], flush=True)

    dump(ROOT / "results" / "qwen_baseline.json", {
        "model": CFG["model"],
        "revision": CFG["revision"],
        "system": CFG["system"],
        "enable_thinking": False,
        "decoding": "greedy",
        "max_new_tokens": CFG["max_new_tokens"],
        "responses": responses,
    })


if __name__ == "__main__":
    main()
