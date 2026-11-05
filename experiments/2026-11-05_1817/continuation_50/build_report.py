"""Combine complete continuation reviews using the explicit 50% style target.

Existing 100 questions are development data. Fresh held-out results are reported
separately. Content correctness never enters either success decision.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
sys.path.insert(0, str(PARENT))
from report import evaluate

TARGET = 0.5
PROTOCOL = {
    "model": "Qwen/Qwen3.5-0.8B",
    "revision": "2fc06364715b967f1860aea9cf38778875588b17",
    "system": "日本語で簡潔に、原則1文で答えてください。",
    "max_new_tokens": 128,
    "do_sample": False,
    "enable_thinking": False,
    "decoding": "greedy",
    "dtype": "float16",
    "batch_size": 8,
}
VERIFICATION_FLAGS = [
    "noncumulative", "restore_hashes_equal", "restore_greedy_equal",
    "all_other_parameters_unchanged", "finite", "reload_hashes_equal",
    "reload_greedy_equal", "embedding_output_final_norm_unchanged",
    "base_weights_tied", "reloaded_weights_tied", "weight_tying_preserved",
    "heldout_generated_after_loading_frozen_selection",
]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path, text=True):
    if text:
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inside(root, value):
    path = (root / value).resolve()
    require(path.is_relative_to(root.resolve()), f"Path escapes {root}: {value}")
    return path


def identity(rows):
    return [(row["id"], row["prompt"], row["category"]) for row in rows]


def canonical_questions(path, expected_count):
    rows = read(path)
    require(len(rows) == expected_count, f"Unexpected question count: {path}")
    require(len({row["id"] for row in rows}) == expected_count, f"Repeated question IDs: {path}")
    require(len({row["prompt"] for row in rows}) == expected_count, f"Repeated prompts: {path}")
    return rows


def check_generation(log, label, continuation=False):
    for key, expected in PROTOCOL.items():
        require(key in log and log[key] == expected, f"{label}: generation setting differs: {key}")
    if continuation:
        require(log.get("mode") == "base" and log.get("diagnostic_only") is False,
                f"{label}: prompted diagnostics cannot be counted as model-edit results")
    else:
        require(log.get("mode", "base") == "base" and log.get("diagnostic_only", False) is False,
                f"{label}: historical result contains a diagnostic prompt")


def reviewed_condition(log_path, review_path, questions, label, continuation=False):
    log, reviews = read(log_path), read(review_path)
    check_generation(log, label, continuation)
    require(identity(log["responses"]) == identity(questions),
            f"{label}: question IDs, wording, category, or order differ")
    result = evaluate(log, reviews)
    require(result["metrics"]["review_complete"], f"{label}: complete per-question reviews are required")
    require(result["metrics"]["reviewed"] == len(questions), f"{label}: missing reviews")
    return result


def compare_50(baseline, edited, role):
    require(identity(baseline["responses"]) == identity(edited["responses"]),
            "Compared conditions use different questions")
    require(all(baseline["generation"][key] == edited["generation"][key] for key in PROTOCOL),
            "Compared conditions use different generation settings")
    require(baseline["metrics"]["review_complete"] and edited["metrics"]["review_complete"],
            "A 50% success decision requires complete reviews")
    before = baseline["metrics"]["style_rate"]
    after = edited["metrics"]["style_rate"]
    return {
        "assessable": True,
        "style_target": TARGET,
        "success_criterion": "Reviewed natural target endings only; content correctness is excluded.",
        "evaluation_role": role,
        "baseline_style_rate": before,
        "edited_style_rate": after,
        "absolute_style_rate_change": after - before,
        "goal_met": after >= TARGET,
    }


def add_counts(condition):
    rows = condition["responses"]
    condition["metrics"]["style_successes"] = sum(row["review"]["style_ok"] for row in rows)
    for category, metrics in condition["by_category"].items():
        metrics["style_successes"] = sum(row["review"]["style_ok"] for row in rows
                                         if row["category"] == category)


def verify_provenance(selection, verification, heldout):
    candidate = selection.get("id", "")
    require(bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", candidate)), "Invalid selected candidate id")
    require(selection.get("target_style_rate") == TARGET, "Selection is not frozen for the 50% target")
    require(selection.get("model") == PROTOCOL["model"] and selection.get("revision") == PROTOCOL["revision"],
            "Selected original model differs from the fixed protocol")
    require(all(selection["generation"].get(key) == value for key, value in PROTOCOL.items()),
            "Frozen generation settings differ from the protocol")
    require(verification.get("selection") == selection, "Verification contains a different frozen selection")
    require(verification.get("selection_sha256") == sha256(ROOT / "results/selection.json"),
            "Selection file differs from its verified SHA-256")
    require(verification.get("model") == selection["model"] and verification.get("revision") == selection["revision"],
            "Verification identifies a different original model")
    for key in VERIFICATION_FLAGS:
        require(verification.get(key) is True, f"Required verification did not pass: {key}")
    for name, expected in selection["file_hashes"].items():
        require(sha256(inside(ROOT, name)) == expected, f"Frozen continuation input changed: {name}")
    for name, expected in selection["shared_file_hashes"].items():
        require(sha256(inside(PARENT, name)) == expected, f"Frozen shared source changed: {name}")

    delta = inside(ROOT, selection["edit_file"])
    require(delta.is_relative_to((ROOT / "models").resolve()), "Delta is outside continuation models")
    require(delta.name == f"{candidate}_delta.safetensors", "Delta filename does not identify the selected candidate")
    require(inside(ROOT, verification["edit_file"]) == delta, "Verification identifies a different delta file")
    require(sha256(delta, text=False) == selection["delta_sha256"] == verification.get("delta_sha256"),
            "Selected delta SHA-256 does not match the saved edit")
    require(sha256(ROOT / "models/actual_delta.safetensors", text=False) == verification.get("actual_delta_sha256"),
            "Actual stored-dtype delta differs from the verified artifact")
    require(verification.get("source_checkpoint_sha256") == selection["expected_source_checkpoint_sha256"],
            "Verified original checkpoint differs from the frozen source")

    heldout_hash = sha256(ROOT / "data/heldout.json")
    require(heldout_hash == selection.get("heldout_sha256") == verification.get("heldout_sha256"),
            "Held-out question file differs from the frozen and verified input")
    require(inside(ROOT, verification["heldout_file"]) == (ROOT / "data/heldout.json").resolve(),
            "Verification refers to another held-out dataset")
    require(verification.get("heldout_n") == len(heldout) == 50, "Verified held-out count differs")
    require(verification.get("probe_n") == 8 and verification.get("probe_ids") == [row["id"] for row in heldout[:8]],
            "Verification probe IDs differ from the held-out first batch")

    before, after = read(ROOT / "results/base_hashes.json"), verification["after_hashes"]
    require(before.keys() == after.keys(), "Parameter names differ before and after editing")
    changed = verification["changed_elements"]
    allowed = {f"model.layers.{layer}.mlp.down_proj.weight" for layer in selection["spec"]["layers"]}
    require(set(changed) == allowed and all(type(count) is int and count >= 0 for count in changed.values()),
            "Changed-parameter record differs from the selected internal projections")
    actual_changed = {name for name in before if before[name] != after[name]}
    require(actual_changed == {name for name, count in changed.items() if count > 0} and bool(actual_changed),
            "Parameter hash changes do not match the verified nonzero edit")
    model_path = inside(ROOT, verification["model_directory"])
    require(model_path == (ROOT / "models/zundamon-edited").resolve() and model_path.is_dir(),
            "Verified standalone model directory is missing or differs")
    return heldout_hash


def build_report():
    selection = read(ROOT / "results/selection.json")
    verification = read(ROOT / "results/verification.json")
    questions = canonical_questions(PARENT / "data/questions.json", 100)
    validation_questions = canonical_questions(PARENT / "data/validation.json", 50)
    heldout_questions = canonical_questions(ROOT / "data/heldout.json", 50)
    require(not ({row["prompt"] for row in heldout_questions} &
                 {row["prompt"] for row in questions + validation_questions}),
            "Fresh held-out prompts overlap the evaluated development sets")
    heldout_hash = verify_provenance(selection, verification, heldout_questions)
    candidate = selection["id"]
    paths = {
        "base": (PARENT / "results/base_questions_128.json", PARENT / "reviews/base_questions.json"),
        "previous_edited": (PARENT / "results/edited_questions_128.json", PARENT / "reviews/edited_questions.json"),
        "edited": (ROOT / f"results/questions_{candidate}.json", ROOT / f"reviews/questions_{candidate}.json"),
    }
    conditions = {name: reviewed_condition(log, reviews, questions, name, continuation=name == "edited")
                  for name, (log, reviews) in paths.items()}
    validation_paths = (ROOT / f"results/validation_{candidate}.json", ROOT / f"reviews/validation_{candidate}.json")
    validation = reviewed_condition(*validation_paths, validation_questions, "validation", continuation=True)
    for split, result in [("questions", conditions["edited"]), ("validation", validation)]:
        require(result["metrics"] == selection["reviewed_development_metrics"][split],
                f"Recomputed {split} metrics differ from the frozen selection")

    heldout_paths = {
        "base": (ROOT / "results/base_heldout_128.json", ROOT / "reviews/base_heldout.json"),
        "edited": (ROOT / "results/edited_heldout_128.json", ROOT / "reviews/edited_heldout.json"),
    }
    heldout_conditions = {name: reviewed_condition(log, reviews, heldout_questions, f"heldout {name}", continuation=True)
                          for name, (log, reviews) in heldout_paths.items()}
    for name, result in heldout_conditions.items():
        generation = result["generation"]
        require(generation.get("condition") == name and generation.get("heldout_sha256") == heldout_hash,
                f"Held-out {name} generation metadata differs from verification")
        if name == "edited":
            require(generation.get("selection_id") == candidate and generation.get("delta_sha256") == selection["delta_sha256"],
                    "Held-out edited generation identifies a different selection")

    comparison = compare_50(conditions["base"], conditions["edited"], "development: existing 100 questions")
    heldout_comparison = compare_50(heldout_conditions["base"], heldout_conditions["edited"],
                                   "fresh held-out: generated after candidate freeze")
    for condition in [*conditions.values(), validation, *heldout_conditions.values()]:
        add_counts(condition)
    inputs = {f"{name}_{kind}": {"path": path.relative_to(PARENT).as_posix(), "sha256": sha256(path)}
              for name, pair in {**paths, "validation": validation_paths,
                                 **{f"heldout_{key}": value for key, value in heldout_paths.items()}}.items()
              for kind, path in zip(["log", "review"], pair)}
    return {
        "review_notice": "Agent reviews of natural target endings; content correctness never affects acceptance.",
        "target_style_rate": TARGET,
        "evaluation_role": "The existing 100 questions and validation are development sets, not unseen tests.",
        "previous_edited_notice": "Historical 1817 result is a reference only; the continuation starts from original weights.",
        "conditions": conditions,
        "comparison": comparison,
        "validation": {"evaluation_role": "development: candidate comparison", **validation},
        "heldout": {"conditions": heldout_conditions, "comparison": heldout_comparison},
        "provenance": {
            "selected_id": candidate, "selection_sha256": sha256(ROOT / "results/selection.json"),
            "verification_sha256": sha256(ROOT / "results/verification.json"),
            "delta_sha256": selection["delta_sha256"], "heldout_sha256": heldout_hash,
            "source_checkpoint_sha256": verification["source_checkpoint_sha256"],
            "verified_changed_elements": verification["changed_elements"],
            "selection_and_verification_consistent": True,
            "report_builder_sha256": sha256(Path(__file__)),
            "aggregation_source_sha256": sha256(PARENT / "report.py"),
            "text_hash_policy": "UTF-8 with LF line endings; binary deltas hashed without normalization",
            "inputs": inputs,
        },
    }


def main(output="results/report.json"):
    destination = inside(ROOT / "results", Path(output).relative_to("results"))
    result = build_report()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({
        "target_style_rate": TARGET,
        "existing_100": {key: value["metrics"] for key, value in result["conditions"].items()},
        "existing_100_comparison": result["comparison"],
        "validation": result["validation"]["metrics"],
        "fresh_heldout_50": {key: value["metrics"] for key, value in result["heldout"]["conditions"].items()},
        "heldout_comparison": result["heldout"]["comparison"],
        "report": str(destination),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/report.json", help="Destination inside continuation_50/results")
    main(parser.parse_args().output)
