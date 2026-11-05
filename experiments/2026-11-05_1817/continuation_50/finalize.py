"""Save a frozen continuation edit and evaluate fresh held-out questions.

All writes stay inside continuation_50. Existing finalization artifacts are
rejected rather than overwritten. The original 1817 outputs are read-only.
"""
import argparse
import gc
import hashlib
import json
import platform
import re
from pathlib import Path

import run as r
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
SYSTEM = "日本語で簡潔に、原則1文で答えてください。"
ALLOWED_PARAMETER = re.compile(r"model\.layers\.\d+\.mlp\.down_proj\.weight")


def sha256(path, text=False):
    if text:
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def local_path(value, subtree=None):
    path = (ROOT / value).resolve()
    boundary = (ROOT / subtree).resolve() if subtree else ROOT.resolve()
    if not path.is_relative_to(boundary):
        raise ValueError(f"Path must stay inside {boundary}: {value}")
    return path


def read_selection(selection_path, edit_override=None):
    selection_path = local_path(selection_path, "results")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    candidate = selection.get("id", "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", candidate):
        raise ValueError("Selection requires a safe nonempty candidate id")
    expected = selection.get("delta_sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Selection requires the SHA-256 of the frozen delta")
    if selection.get("model", MODEL) != MODEL or selection.get("revision", REVISION) != REVISION:
        raise ValueError("Selection does not use the fixed original checkpoint")
    declared = selection.get("edit_file", f"models/{candidate}_delta.safetensors")
    edit_path = local_path(declared, "models")
    if edit_override is not None and local_path(edit_override, "models") != edit_path:
        raise ValueError("--edit differs from the file identified by the selection")
    if edit_path.name != f"{candidate}_delta.safetensors":
        raise ValueError("Delta filename does not match the selected candidate id")
    if sha256(edit_path) != expected:
        raise ValueError("Delta bytes differ from the frozen selection")
    return selection, selection_path, edit_path


def validate_protocol():
    cfg = r.e.CFG
    expected = {"model": MODEL, "revision": REVISION, "system": SYSTEM,
                "max_new_tokens": 128, "enable_thinking": False, "do_sample": False}
    if any(cfg.get(key) != value for key, value in expected.items()):
        raise ValueError("Shared configuration differs from the fixed protocol")
    if r.messages("probe", mode="base") != [
        {"role": "system", "content": SYSTEM}, {"role": "user", "content": "probe"}
    ]:
        raise ValueError("Evaluation must use only the neutral system and user question")


def annotate(result, path, selection, condition, heldout_hash):
    result.update(condition=condition, heldout_sha256=heldout_hash,
                  text_hash_policy="UTF-8 with LF line endings")
    if condition == "edited":
        result.update(selection_id=selection["id"], delta_sha256=selection["delta_sha256"])
    r.e.dump(path, result)


@torch.inference_mode()
def main(selection_path="results/selection.json", edit_override=None):
    validate_protocol()
    selection, selection_path, edit_path = read_selection(selection_path, edit_override)
    heldout_path = ROOT / "data/heldout.json"
    heldout_hash = sha256(heldout_path, text=True)
    if selection.get("heldout_sha256", heldout_hash) != heldout_hash:
        raise ValueError("Held-out input differs from the frozen selection")
    questions = json.loads(heldout_path.read_text(encoding="utf-8"))
    if len(questions) != 50 or len({row["id"] for row in questions}) != 50:
        raise ValueError("Held-out evaluation requires exactly 50 unique question IDs")
    if len({row["prompt"] for row in questions}) != 50:
        raise ValueError("Held-out evaluation contains repeated prompts")

    folder = ROOT / "models/zundamon-edited"
    actual_path = ROOT / "models/actual_delta.safetensors"
    base_path = ROOT / "results/base_heldout_128.json"
    edited_path = ROOT / "results/edited_heldout_128.json"
    hashes_path = ROOT / "results/base_hashes.json"
    verification_path = ROOT / "results/verification.json"
    probe_directory = ROOT / "runs/finalization"
    destinations = [folder, actual_path, base_path, edited_path, hashes_path,
                    verification_path, probe_directory]
    occupied = [str(path.relative_to(ROOT)) for path in destinations if path.exists()]
    if occupied:
        raise FileExistsError(f"Finalization outputs already exist: {occupied}")
    for directory in [ROOT / "models", ROOT / "results", ROOT / "runs"]:
        directory.mkdir(exist_ok=True)
    # Reserve both destinations before loading the model or producing any answers.
    folder.mkdir(exist_ok=False)
    probe_directory.mkdir(exist_ok=False)

    edits = load_file(str(edit_path))
    if not edits or any(not ALLOWED_PARAMETER.fullmatch(name) for name in edits):
        raise ValueError("Only internal MLP down_proj weights may be edited")
    source = Path(hf_hub_download(MODEL, "model.safetensors-00001-of-00001.safetensors",
                                 revision=REVISION, local_files_only=True))
    source_hash = sha256(source)
    if selection.get("expected_source_checkpoint_sha256", source_hash) != source_hash:
        raise ValueError("Original checkpoint SHA differs from the frozen source")
    model, tok = r.e.load_model()
    base_tied = model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
    if not base_tied or not model.config.tie_word_embeddings:
        raise AssertionError("Original checkpoint must retain tied input/output embeddings")
    before = r.e.hashes(model)
    originals = {name: model.get_parameter(name).detach().cpu().clone() for name in edits}
    base = r.generate(model, tok, questions, base_path, mode="base")
    annotate(base, base_path, selection, "base", heldout_hash)
    probe_questions = questions[:8]
    base_probe = base["responses"][:8]

    r.e.apply(model, originals, edits)
    after = r.e.hashes(model)
    r.e.apply(model, originals, edits)
    assert r.e.hashes(model) == after, "Repeated application accumulated the edit"
    actual = {name: model.get_parameter(name).detach().cpu().float() - original.float()
              for name, original in originals.items()}
    changed = {name: int(torch.count_nonzero(delta)) for name, delta in actual.items()}
    assert any(changed.values()), "A zero delta is not an edited model"
    assert all(after[name] == digest for name, digest in before.items() if name not in edits)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    assert model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()

    r.e.apply(model, originals, {})
    assert r.e.hashes(model) == before, "Restoration did not recover original parameters"
    restored = r.generate(model, tok, probe_questions, probe_directory / "restored_probe.json", mode="base")
    assert restored["responses"] == base_probe, "Restored model generated different probe answers"
    r.e.apply(model, originals, edits)
    assert r.e.hashes(model) == after
    model.save_pretrained(folder)
    tok.save_pretrained(folder)
    save_file(actual, str(actual_path))
    edited = r.generate(model, tok, questions, edited_path, mode="base")
    annotate(edited, edited_path, selection, "edited", heldout_hash)
    edited_probe = edited["responses"][:8]
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    reloaded, reloaded_tok = r.e.load_model(str(folder))
    assert r.e.hashes(reloaded) == after, "Saved checkpoint parameter hashes differ"
    reload_tied = reloaded.model.embed_tokens.weight.data_ptr() == reloaded.lm_head.weight.data_ptr()
    assert reload_tied == base_tied and reloaded.config.tie_word_embeddings == base_tied
    probe = r.generate(reloaded, reloaded_tok, probe_questions,
                       probe_directory / "reloaded_probe.json", mode="base")
    assert probe["responses"] == edited_probe, "Reloaded model generated different probe answers"
    r.e.dump(hashes_path, before)
    r.e.dump(verification_path, {
        "model": MODEL, "revision": REVISION, "source_checkpoint_sha256": source_hash,
        "selection": selection, "selection_sha256": sha256(selection_path, text=True),
        "edit_file": edit_path.relative_to(ROOT).as_posix(),
        "delta_sha256": selection["delta_sha256"], "actual_delta_sha256": sha256(actual_path),
        "model_directory": folder.relative_to(ROOT).as_posix(),
        "heldout_file": "data/heldout.json", "heldout_sha256": heldout_hash,
        "heldout_n": len(questions), "heldout_generated_after_loading_frozen_selection": True,
        "probe_ids": [row["id"] for row in probe_questions], "probe_n": len(probe_questions),
        "changed_elements": changed, "noncumulative": True,
        "restore_hashes_equal": True, "restore_greedy_equal": True,
        "all_other_parameters_unchanged": True, "finite": True,
        "reload_hashes_equal": True, "reload_greedy_equal": True,
        "embedding_output_final_norm_unchanged": True,
        "base_weights_tied": base_tied, "reloaded_weights_tied": reload_tied,
        "weight_tying_preserved": base_tied == reload_tied,
        "after_hashes": after, "torch_version": torch.__version__,
        "python": platform.python_version(), "gpu": torch.cuda.get_device_name(),
        "text_hash_policy": "UTF-8 with LF line endings",
    })
    print(json.dumps({"saved_model": str(folder), "selected_id": selection["id"],
                      "changed_elements": changed, "heldout_n": len(questions),
                      "verification": str(verification_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", default="results/selection.json")
    parser.add_argument("--edit", help="Optional consistency check against selection.edit_file")
    args = parser.parse_args()
    main(args.selection, args.edit)
