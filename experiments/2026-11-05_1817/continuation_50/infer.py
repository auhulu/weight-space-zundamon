"""Ask the saved continuation model a question; log the unmodified answer."""
import argparse
import uuid
from pathlib import Path

import run as r

ROOT = Path(__file__).resolve().parent


def local_path(value, subtree):
    path = (ROOT / value).resolve()
    if not path.is_relative_to((ROOT / subtree).resolve()):
        raise ValueError(f"Path must stay inside continuation_50/{subtree}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--model", default="models/zundamon-edited")
    parser.add_argument("--output", help="Optional new log path inside continuation_50/runs")
    args = parser.parse_args()
    model_path = local_path(args.model, "models")
    output = local_path(args.output or f"runs/inference/{uuid.uuid4().hex}.json", "runs")
    if output.exists():
        raise FileExistsError(f"Inference log already exists: {output}")
    if not model_path.is_dir():
        raise FileNotFoundError(f"Saved model not found: {model_path}")
    if r.messages(args.prompt, mode="base") != [
        {"role": "system", "content": "日本語で簡潔に、原則1文で答えてください。"},
        {"role": "user", "content": args.prompt},
    ]:
        raise ValueError("Inference must use only the neutral system and user question")
    model, tok = r.e.load_model(str(model_path))
    result = r.generate(model, tok, [{"id": "interactive", "category": "user", "prompt": args.prompt}],
                        output, mode="base")
    result["checkpoint"] = model_path.relative_to(ROOT).as_posix()
    r.e.dump(output, result)
    print(result["responses"][0]["answer"])


if __name__ == "__main__":
    main()
