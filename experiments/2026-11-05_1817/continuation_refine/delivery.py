"""Freeze, save, report, and rebuild a refinement of the 56% development result.

Only this continuation directory is writable. Historical artifacts are inputs.
All success decisions use reviewed natural endings, without content correctness.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import uuid

import research as r
import torch
from safetensors.torch import load_file, save_file
from report import evaluate

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
PREVIOUS = PARENT / "continuation_50"
TARGET = 0.56
PROTOCOL = {
    "model": "Qwen/Qwen3.5-0.8B",
    "revision": "2fc06364715b967f1860aea9cf38778875588b17",
    "system": "日本語で簡潔に、原則1文で答えてください。",
    "max_new_tokens": 128, "batch_size": 8, "enable_thinking": False,
    "do_sample": False, "decoding": "greedy", "dtype": "float16",
}
FLAGS = [
    "noncumulative", "restore_hashes_equal", "restore_greedy_equal",
    "all_other_parameters_unchanged", "finite", "reload_hashes_equal",
    "reload_greedy_equal", "embedding_output_final_norm_unchanged",
    "base_weights_tied", "reloaded_weights_tied", "weight_tying_preserved",
    "heldout_generated_after_loading_frozen_selection",
]
PARAMETER = re.compile(r"model\.layers\.\d+\.mlp\.down_proj\.weight")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path, text=True):
    if text:
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inside(root, name):
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()), f"Path escapes {root}: {name}")
    return path


def safe_name(name):
    require(isinstance(name, str) and bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)),
            "Candidate/family must be a safe nonempty name")
    return name


def write_new(path, data):
    inside(ROOT, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)


def identity(rows):
    return [(x["id"], x["prompt"], x["category"]) for x in rows]


def questions(path, count):
    rows = read(path)
    require(len(rows) == count and len({x["id"] for x in rows}) == count,
            f"Expected {count} unique question IDs: {path}")
    require(len({x["prompt"] for x in rows}) == count, f"Duplicate prompts: {path}")
    return rows


def protocol(log, label, new=False):
    require(all(log.get(k) == v for k, v in PROTOCOL.items()), f"Generation settings differ: {label}")
    require(log.get("mode", None if new else "base") == "base" and
            log.get("diagnostic_only", None if new else False) is False,
            f"Diagnostic prompts are not evaluation: {label}")


def reviewed(log_path, review_path, canonical, new=False):
    log = read(log_path)
    protocol(log, str(log_path), new)
    require(identity(log["responses"]) == identity(canonical), f"Question wording/order differs: {log_path}")
    result = evaluate(log, read(review_path))
    require(result["metrics"]["review_complete"] and result["metrics"]["reviewed"] == len(canonical),
            f"Complete reasoned reviews required: {review_path}")
    return result


def heldout_questions():
    heldout = questions(ROOT / "data/heldout.json", 50)
    prompts = {x["prompt"] for x in heldout}
    other_paths = [PARENT / "data/questions.json", PARENT / "data/validation.json",
                   PARENT / "data/generalization.json", PARENT / "data/train_pairs.json",
                   PREVIOUS / "data/heldout.json", PREVIOUS / "data/pairs.json"]
    other_paths.extend(path for path in (ROOT / "data").glob("*.json") if path.name != "heldout.json")
    for path in other_paths:
        data = read(path)
        if isinstance(data, list) and all(isinstance(x, dict) and "prompt" in x for x in data):
            require(not prompts.intersection(x["prompt"] for x in data), f"Held-out prompt overlap: {path}")
    require(not prompts.intersection(x[0] for x in r.prior.EXAMPLES), "Held-out prompts overlap few-shot examples")
    return heldout


def development(candidate, delta_hash):
    result = {}
    for split, count in [("questions", 100), ("validation", 50)]:
        canonical = questions(PARENT / f"data/{split}.json", count)
        item = reviewed(ROOT / f"results/{split}_{candidate}.json",
                        ROOT / f"reviews/{split}_{candidate}.json", canonical, new=True)
        require(item["generation"].get("candidate") == candidate and
                item["generation"].get("delta_sha256") == delta_hash,
                f"Development log is not the selected delta: {split}")
        result[split] = item
    return result


def freeze(candidate, family):
    candidate, family = safe_name(candidate), safe_name(family)
    destination = ROOT / "results/selection.json"
    if destination.exists():
        raise FileExistsError(destination)
    for name in ["base_heldout_128.json", "edited_heldout_128.json", "previous_heldout_128.json"]:
        require(not (ROOT / "results" / name).exists(), "Held-out generation must follow selection freeze")
    protocol({**r.e.CFG, "mode": "base", "diagnostic_only": False}, "shared configuration", new=True)
    heldout_questions()
    records = read(ROOT / f"results/{family}_search.json")
    matches = [x for x in records if x["id"] == candidate]
    require(len(matches) == 1, "Search must contain exactly one selected candidate")
    record = matches[0]
    delta_path = ROOT / f"models/{candidate}_delta.safetensors"
    delta_hash = sha(delta_path, text=False)
    require(record.get("delta_sha256") == delta_hash, "Search delta SHA differs from candidate artifact")
    edits = load_file(str(delta_path))
    require(bool(edits) and all(PARAMETER.fullmatch(name) and torch.isfinite(value).all()
                                for name, value in edits.items()), "Invalid internal-projection delta")
    current = development(candidate, delta_hash)
    old = reviewed(PREVIOUS / "results/questions_branch_04.json", PREVIOUS / "reviews/questions_branch_04.json",
                   questions(PARENT / "data/questions.json", 100), new=True)
    require(old["metrics"]["style_rate"] == TARGET, "Previous reference is not the recorded 56% model")
    require(current["questions"]["metrics"]["style_rate"] > TARGET,
            "Refinement must exceed 56/100, requiring at least 57 reviewed successes")
    previous_verification = read(PREVIOUS / "results/verification.json")
    require(previous_verification["model"] == PROTOCOL["model"] and
            previous_verification["revision"] == PROTOCOL["revision"], "Previous source identity differs")
    local = sorted({str(path.relative_to(ROOT).as_posix()) for path in ROOT.glob("*.py")} |
                   {str(path.relative_to(ROOT).as_posix()) for path in (ROOT / "data").glob("*.json")} |
                   {f"results/{family}_search.json", f"results/questions_{candidate}.json",
                    f"results/validation_{candidate}.json", f"reviews/questions_{candidate}.json",
                    f"reviews/validation_{candidate}.json"})
    shared = ["config.json", "experiment.py", "scoring.py", "report.py", "data/questions.json",
              "data/validation.json", "continuation_50/run.py", "continuation_50/finalize.py",
              "continuation_50/data/pairs.json", "continuation_50/results/selection.json",
              "continuation_50/results/verification.json", "continuation_50/results/questions_branch_04.json",
              "continuation_50/reviews/questions_branch_04.json", "results/base_questions_128.json",
              "reviews/base_questions.json"]
    selection = {
        "id": candidate, "family": family, "spec": record["spec"], "details": record["details"],
        "model": PROTOCOL["model"], "revision": PROTOCOL["revision"], "generation": dict(PROTOCOL),
        "edit_file": delta_path.relative_to(ROOT).as_posix(), "delta_sha256": delta_hash,
        "edited_parameter_names": sorted(edits),
        "heldout_sha256": sha(ROOT / "data/heldout.json"),
        "expected_source_checkpoint_sha256": previous_verification["source_checkpoint_sha256"],
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_style_rate": TARGET, "target_comparison": ">", "content_is_acceptance_criterion": False,
        "reviewed_development_metrics": {key: value["metrics"] for key, value in current.items()},
        "previous_reference_style_rate": old["metrics"]["style_rate"],
        "evaluation_role": "Existing 100 and validation 50 are development; fresh held-out generation follows freeze.",
        "selection_reason": "Reviewed natural-ending success exceeds the previous 56/100; content correctness is excluded.",
        "file_hashes": {name: sha(ROOT / name) for name in local},
        "shared_file_hashes": {name: sha(PARENT / name) for name in shared},
        "observation_file": "continuation_50/runs/fewshot_stats.pt",
        "observation_sha256": sha(PREVIOUS / "runs/fewshot_stats.pt", text=False),
        "text_hash_policy": "UTF-8 with LF line endings; binary artifacts are not normalized",
    }
    write_new(destination, selection)
    print(json.dumps({"selected": candidate, "delta_sha256": delta_hash,
                      "metrics": selection["reviewed_development_metrics"]}, ensure_ascii=False, indent=2))


def frozen(need_delta=True):
    selection = read(ROOT / "results/selection.json")
    candidate = safe_name(selection["id"])
    require(selection.get("target_style_rate") == TARGET and selection.get("target_comparison") == ">",
            "Selection does not use the strict 56% improvement goal")
    require(selection.get("model") == PROTOCOL["model"] and selection.get("revision") == PROTOCOL["revision"],
            "Selection original checkpoint differs")
    require(selection.get("generation") == PROTOCOL, "Frozen generation settings differ")
    require(selection.get("content_is_acceptance_criterion") is False, "Content must not enter acceptance")
    for name, expected in selection["file_hashes"].items():
        require(sha(inside(ROOT, name)) == expected, f"Frozen input/source differs: {name}")
    for name, expected in selection["shared_file_hashes"].items():
        require(sha(inside(PARENT, name)) == expected, f"Frozen historical input/source differs: {name}")
    require(sha(inside(PARENT, selection["observation_file"]), text=False) == selection["observation_sha256"],
            "Frozen observation tensor differs")
    require(sha(ROOT / "data/heldout.json") == selection["heldout_sha256"], "Frozen held-out questions differ")
    heldout_questions()
    delta_path = inside(ROOT / "models", Path(selection["edit_file"]).relative_to("models"))
    require(delta_path.name == f"{candidate}_delta.safetensors", "Delta filename differs from selected ID")
    if need_delta:
        require(sha(delta_path, text=False) == selection["delta_sha256"], "Selected delta SHA differs")
        require(sorted(load_file(str(delta_path))) == selection["edited_parameter_names"], "Delta parameter names differ")
    current = development(candidate, selection["delta_sha256"])
    require({key: value["metrics"] for key, value in current.items()} == selection["reviewed_development_metrics"],
            "Frozen reviewed development metrics differ")
    require(current["questions"]["metrics"]["style_rate"] > TARGET, "Frozen candidate misses strict improvement")
    return selection, delta_path


def previous_finalizer():
    spec = importlib.util.spec_from_file_location("refine_previous_finalizer", PREVIOUS / "finalize.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    # This module's functions write only explicit paths derived from ROOT.
    # Its run module supplies neutral generation and original-checkpoint loading.
    require(module.r is r.prior, "Unexpected run module used by reused finalizer")
    return module


@torch.inference_mode()
def finalize():
    selection, _ = frozen()
    output = ROOT / "results/previous_heldout_128.json"
    if output.exists():
        raise FileExistsError(output)
    previous_finalizer().main()
    gc.collect()
    torch.cuda.empty_cache()
    old_verification = read(PREVIOUS / "results/verification.json")
    old_selection = read(PREVIOUS / "results/selection.json")
    model, tokenizer = r.e.load_model(str(PREVIOUS / "models/zundamon-edited"))
    loaded_hashes = r.e.hashes(model)
    require(loaded_hashes == old_verification["after_hashes"], "Previous 56% saved model hashes differ")
    require(model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr(),
            "Previous model input/output embeddings are not tied")
    result = r.prior.generate(model, tokenizer, heldout_questions(), output, mode="base")
    result.update(condition="previous", heldout_sha256=selection["heldout_sha256"],
                  selection_id=old_selection["id"], delta_sha256=old_selection["delta_sha256"],
                  current_selection_sha256=sha(ROOT / "results/selection.json"),
                  previous_verification_sha256=sha(PREVIOUS / "results/verification.json"),
                  previous_model_hashes_verified=True,
                  checkpoint="../continuation_50/models/zundamon-edited")
    r.e.dump(output, result)
    print(json.dumps({"previous_heldout": str(output), "previous_model_hashes_verified": True}, indent=2))


def validate_verification(selection, delta_path):
    verification = read(ROOT / "results/verification.json")
    require(verification.get("selection") == selection and
            verification.get("selection_sha256") == sha(ROOT / "results/selection.json"),
            "Verification belongs to a different frozen selection")
    require(verification.get("model") == PROTOCOL["model"] and
            verification.get("revision") == PROTOCOL["revision"], "Verified source identity differs")
    require(all(verification.get(key) is True for key in FLAGS), "One or more required save/reload checks failed")
    require(verification.get("delta_sha256") == selection["delta_sha256"] and
            inside(ROOT, verification["edit_file"]) == delta_path, "Verified edit differs")
    require(sha(ROOT / "models/actual_delta.safetensors", text=False) == verification["actual_delta_sha256"],
            "Actual FP16 delta bytes differ")
    require(verification.get("source_checkpoint_sha256") == selection["expected_source_checkpoint_sha256"],
            "Verified original checkpoint SHA differs")
    heldout = heldout_questions()
    require(verification.get("heldout_sha256") == selection["heldout_sha256"] and
            verification.get("heldout_n") == 50 and verification.get("heldout_file") == "data/heldout.json",
            "Verified held-out dataset differs")
    require(verification.get("probe_n") == 8 and verification.get("probe_ids") == [x["id"] for x in heldout[:8]],
            "Verified probe questions differ")
    before, after, changed = read(ROOT / "results/base_hashes.json"), verification["after_hashes"], verification["changed_elements"]
    require(before.keys() == after.keys(), "Parameter names changed")
    require(set(changed) == set(selection["edited_parameter_names"]) and
            all(PARAMETER.fullmatch(name) and type(count) is int and count >= 0 for name, count in changed.items()),
            "Changed parameter record differs from selected internal projections")
    require({name for name in before if before[name] != after[name]} ==
            {name for name, count in changed.items() if count > 0} and any(changed.values()),
            "Observed hash changes differ from nonzero selected edits")
    model_path = inside(ROOT, verification["model_directory"])
    require(model_path == ROOT / "models/zundamon-edited" and model_path.is_dir(), "Saved model directory differs")
    return verification


def counts(condition):
    rows = condition["responses"]
    condition["metrics"]["style_successes"] = sum(x["review"]["style_ok"] for x in rows)
    for category, metric in condition["by_category"].items():
        metric["style_successes"] = sum(x["review"]["style_ok"] for x in rows if x["category"] == category)


def comparison(conditions, development_set):
    reference = conditions["base"]
    for item in conditions.values():
        require(identity(item["responses"]) == identity(reference["responses"]), "Compared question identities differ")
        require(all(item["generation"][key] == reference["generation"][key] for key in PROTOCOL),
                "Compared generation settings differ")
        require(item["metrics"]["review_complete"], "Comparison requires complete reviews")
    rates = {name: item["metrics"]["style_rate"] for name, item in conditions.items()}
    return {
        "assessable": True, "evaluation_role": "development" if development_set else "fresh held-out after freeze",
        "success_criterion": "Reviewed natural target endings only; content correctness is excluded.",
        "style_rates": rates, "change_vs_base": rates["edited"] - rates["base"],
        "change_vs_previous": rates["edited"] - rates["previous"],
        "improved_vs_previous": rates["edited"] > rates["previous"],
        "target_style_rate": TARGET if development_set else None,
        "target_comparison": ">" if development_set else None,
        "goal_met": rates["edited"] > TARGET if development_set else None,
    }


def build_report():
    selection, delta_path = frozen()
    verification = validate_verification(selection, delta_path)
    candidate = selection["id"]
    canonical = questions(PARENT / "data/questions.json", 100)
    paths = {
        "base": (PARENT / "results/base_questions_128.json", PARENT / "reviews/base_questions.json"),
        "previous": (PREVIOUS / "results/questions_branch_04.json", PREVIOUS / "reviews/questions_branch_04.json"),
        "edited": (ROOT / f"results/questions_{candidate}.json", ROOT / f"reviews/questions_{candidate}.json"),
    }
    current = development(candidate, selection["delta_sha256"])
    conditions = {name: reviewed(log, review, canonical, new=name != "base") for name, (log, review) in paths.items()}
    heldout = {}
    inputs = {f"questions_{name}": pair for name, pair in paths.items()}
    old_selection = read(PREVIOUS / "results/selection.json")
    for name in ["base", "previous", "edited"]:
        log_path, review_path = ROOT / f"results/{name}_heldout_128.json", ROOT / f"reviews/{name}_heldout.json"
        item = reviewed(log_path, review_path, heldout_questions(), new=True)
        generation = item["generation"]
        require(generation.get("condition") == name and generation.get("heldout_sha256") == selection["heldout_sha256"],
                f"Held-out condition identity differs: {name}")
        if name == "edited":
            require(generation.get("selection_id") == candidate and generation.get("delta_sha256") == selection["delta_sha256"],
                    "Held-out edited output belongs to another candidate")
        if name == "previous":
            require(generation.get("selection_id") == old_selection["id"] and
                    generation.get("delta_sha256") == old_selection["delta_sha256"] and
                    generation.get("previous_model_hashes_verified") is True and
                    generation.get("previous_verification_sha256") == sha(PREVIOUS / "results/verification.json") and
                    generation.get("current_selection_sha256") == sha(ROOT / "results/selection.json"),
                    "Previous-model held-out output has mismatched provenance")
        heldout[name] = item
        inputs[f"heldout_{name}"] = (log_path, review_path)
    inputs["validation"] = (ROOT / f"results/validation_{candidate}.json", ROOT / f"reviews/validation_{candidate}.json")
    compared = comparison(conditions, True)
    heldout_compared = comparison(heldout, False)
    for item in [*conditions.values(), *heldout.values(), current["validation"]]:
        counts(item)
    return {
        "review_notice": "Codex per-answer natural-ending reviews; content correctness is not an acceptance criterion.",
        "goal": {"dataset": "existing 100 development questions", "target_style_rate": TARGET,
                 "comparison": ">", "minimum_successes": 57, "goal_met": compared["goal_met"]},
        "conditions": conditions, "comparison": compared,
        "validation": {"evaluation_role": "development: used in candidate selection", **current["validation"]},
        "heldout": {"conditions": heldout, "comparison": heldout_compared},
        "provenance": {"selection": selection, "selection_sha256": sha(ROOT / "results/selection.json"),
                       "verification_sha256": sha(ROOT / "results/verification.json"),
                       "changed_elements": verification["changed_elements"],
                       "selection_and_verification_consistent": True,
                       "inputs": {label: [{"path": path.relative_to(PARENT).as_posix(), "sha256": sha(path)}
                                          for path in pair] for label, pair in inputs.items()}},
    }


def report():
    destination = ROOT / "results/report.json"
    if destination.exists():
        raise FileExistsError(destination)
    result = build_report()
    write_new(destination, result)
    print(json.dumps({"goal": result["goal"], "existing_100": result["comparison"],
                      "validation": result["validation"]["metrics"], "fresh_50": result["heldout"]["comparison"],
                      "report": str(destination)}, ensure_ascii=False, indent=2))


def rebuild():
    selection, destination = frozen(need_delta=False)
    edits, _ = r.make_edit(r.stats(), selection["spec"])
    require(sorted(edits) == selection["edited_parameter_names"], "Rebuilt parameter names differ")
    folder = ROOT / "runs/rebuild"
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / f"{uuid.uuid4().hex}.safetensors"
    save_file(edits, str(temporary))
    require(sha(temporary, text=False) == selection["delta_sha256"], "Reconstructed delta SHA differs")
    if destination.exists():
        require(sha(destination, text=False) == selection["delta_sha256"], "Existing delta differs; refusing to overwrite")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(temporary.read_bytes())
    print(json.dumps({"delta": str(destination), "sha256": sha(destination, text=False), "match": True}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "finalize", "report", "rebuild"])
    parser.add_argument("--candidate")
    parser.add_argument("--family")
    args = parser.parse_args()
    if args.stage == "freeze":
        if not args.candidate or not args.family:
            parser.error("freeze requires --candidate and --family")
        freeze(args.candidate, args.family)
    else:
        {"finalize": finalize, "report": report, "rebuild": rebuild}[args.stage]()


if __name__ == "__main__":
    main()
