"""Evaluate saved generations with Jev without exposing their model labels."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

MODEL = "typesafe/jev-1.14"
INSTRUCTIONS = (
    "対象の回答を書き手自身の発話として読み、文末の語尾だけがずんだもんの口調として"
    "適切か判定してください。内容の正誤、質問への関連性、語尾以外の文法の誤り、"
    "文章の反復や不自然さは採点から外してください。"
    "「〜なのだ」「〜のだ」「〜なのだ？」などは、語尾として機能していれば適切です。"
    "「〜なのだよ」で終わる場合は不適切です。引用中にしか出ない表現や文中だけに出る表現は数えません。"
)
CRITERIA = {
    "appropriate": "最後の発話がずんだもんらしい語尾で終わっている。",
    "inappropriate": "最後の発話が「なのだよ」で終わるか、ずんだもんらしい語尾がないか、語尾自体の接続が不自然。",
    "no_ending": "空欄や書きかけなどで評価できる文末がない。",
}


def judge(texts):
    request = Request(
        "https://openrouter.ai/api/alpha/decisions",
        data=json.dumps({
            "model": MODEL,
            "state": texts,
            "questions": {
                key: {
                    "type": "choice",
                    "instructions": f"`{key}` のみを評価してください。{INSTRUCTIONS}",
                    "criteria": CRITERIA,
                }
                for key in texts
            },
        }, ensure_ascii=False).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        },
    )
    for attempt in range(4):
        try:
            with urlopen(request, timeout=60) as response:
                answers = json.load(response)["answers"]
            return [answers[key] for key in texts]
        except HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt == 3:
                raise RuntimeError(f"Jev HTTP {error.code}: {error.read().decode()}") from error
        except (URLError, TimeoutError):
            if attempt == 3:
                raise
        time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    datasets = [
        json.loads(path.read_text(encoding="utf-8"))["responses"]
        for path in args.inputs
    ]
    texts = list(dict.fromkeys(row["answer"] for rows in datasets for row in rows))
    batches = [
        {f"answer_{index}": text for index, text in enumerate(texts[start:start + 10])}
        for start in range(0, len(texts), 10)
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        decisions = dict(zip(texts, (answer for batch in pool.map(judge, batches) for answer in batch), strict=True))

    results = []
    for path, rows in zip(args.inputs, datasets, strict=True):
        reviews = [{"id": row["id"], "answer": row["answer"], "jev": decisions[row["answer"]]} for row in rows]
        counts = Counter(row["jev"]["choice"] for row in reviews)
        print(f"{path.stem}: {counts['appropriate']}/{len(rows)} appropriate ({dict(counts)})")
        results.append({"input": str(path), "counts": counts, "responses": reviews})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "model": MODEL, "instructions": INSTRUCTIONS, "criteria": CRITERIA, "results": results,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
