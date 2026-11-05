"""CPU-only, read-only logit-lens diagnostics for existing MLP writers.

No model inference, gradient, regression, parameter mutation, or fitted loss.
The final RMS denominator and intervening layers are deliberately excluded:
these are static logit-numerator projections, not causal logit changes.
"""
import json
from pathlib import Path

import torch
from safetensors import safe_open
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
SNAPSHOT = REPO / "cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots" / REVISION
POSITIONS = ["pre2", "pre1", "start", "early", "middle", "branch", "branch_duplicate", "ending", "period"]


def cosine(a, b):
    return float(a.dot(b) / (a.norm() * b.norm()).clamp_min(1e-12))


def compact_summary(result):
    return {
        "method": "CPU-only static projection of original MLP writer columns onto token-margin readout directions",
        "revision": result["revision"],
        "role": result["role"],
        "used_in_adopted_edit": result["used_in_adopted_edit"],
        "cpu_only": result["cpu_only"],
        "original_weights_read_only": result["original_weights_read_only"],
        "readout_formula": result["readout_formula"],
        "writer_formula": result["writer_formula"],
        "limitations": result["limitations"],
        "detailed_log": "runs/writer_diagnostic.json (not tracked by Git)",
        "layers": {
            layer: {
                name: {
                    "branch_d_style_cosine": round(values["d_style_cosine_by_position"][5], 6),
                    "branch_mlp_net_contribution": round(values["mlp_net_contribution_by_position"][5], 6),
                    "selective_promoter_count": values["promoters"]["selective_count"],
                    "selective_suppressor_count": values["suppressors"]["selective_count"],
                }
                for name, values in margins.items()
            }
            for layer, margins in result["layers"].items()
        },
    }


def main():
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    stats = torch.load(ROOT / "runs/fewshot_stats.pt", map_location="cpu", weights_only=True)
    rows = json.loads((ROOT / "data/pairs.json").read_text(encoding="utf-8"))
    groups = {name: [i for i, row in enumerate(rows) if row["category"].startswith(name)]
              for name in ["noun", "verb", "adjective"]}
    groups["non_noun"] = groups["verb"] + groups["adjective"]
    groups["all"] = list(range(len(rows)))
    tok = Tokenizer.from_file(str(SNAPSHOT / "tokenizer.json"))
    token_ids = {text: tok.encode(text, add_special_tokens=False).ids
                 for text in ["なのだ", "のだ", "です", "だ", "。", "ます"]}
    assert all(len(ids) == 1 for ids in token_ids.values()), token_ids
    safepath = SNAPSHOT / "model.safetensors-00001-of-00001.safetensors"
    with safe_open(str(safepath), framework="pt", device="cpu") as weights:
        prefix = "model.language_model."
        norm_gain = weights.get_tensor(prefix + "norm.weight").float() + 1
        embedding = weights.get_slice(prefix + "embed_tokens.weight")
        readouts = {text: embedding[ids[0]:ids[0] + 1].float().squeeze(0) * norm_gain
                    for text, ids in token_ids.items()}
        margin_specs = {
            "noun_nanoda_desu": ("なのだ", "です", "noun"),
            "noun_nanoda_da": ("なのだ", "だ", "noun"),
            "other_noda_period": ("のだ", "。", "non_noun"),
            "other_noda_masu": ("のだ", "ます", "non_noun"),
        }
        result = {
            "revision": REVISION,
            "cpu_only": True,
            "original_weights_read_only": True,
            "role": "exploratory_diagnostic",
            "used_in_adopted_edit": False,
            "positions": POSITIONS,
            "token_ids": token_ids,
            "readout_formula": "v=(embedding[target]-embedding[competitor])*(1+final_norm.weight)",
            "writer_formula": "c_j=(v/||v|| dot down_proj[:,j])*mean(g_j)",
            "limitations": [
                "Final RMS denominator and later layers excluded; values are static logit-numerator projections, not measured output-logit changes.",
                "All g activations use the same teacher-forced styled answer prefix under the neutral prompt.",
                "branch and branch_duplicate are exactly the same position in all 60 pairs.",
                "Column selectors are empirical activation observations; no model parameter is modified or fitted.",
            ],
            "readout_cosines": {},
            "layers": {},
        }
        for name, (target, competitor, group) in margin_specs.items():
            v = readouts[target] - readouts[competitor]
            result["readout_cosines"][name] = {
                "target_competitor_cosine": cosine(readouts[target], readouts[competitor]),
                "norm": float(v.norm()),
            }
        for layer in range(24):
            down = weights.get_tensor(prefix + f"layers.{layer}.mlp.down_proj.weight").float()
            s = stats[layer]
            out = {}
            for name, (target, competitor, group) in margin_specs.items():
                ids = groups[group]
                v = readouts[target] - readouts[competitor]
                unit = v / v.norm()
                projected = unit @ down
                g = s["g"][ids]
                contributions = g * projected
                means = contributions.mean(0)
                d = s["d_style"][ids].mean(0)
                nuisance = means[[0, 1, 2, 3, 4, 7, 8]].abs().amax(0)
                branch = means[5]
                directions = {}
                for label, sign in [("promoters", 1), ("suppressors", -1)]:
                    signed = sign * branch
                    indices = signed.topk(64).indices
                    # Find existing columns whose branch effect exceeds the largest
                    # nuisance-position mean effect by at least 2x.
                    selective = torch.where(signed > 2 * nuisance, signed, torch.zeros_like(signed))
                    selected = selective.topk(64).indices
                    selected = selected[selective[selected] > 0]
                    directions[label] = {
                        "all_positive_mass": float(signed.clamp_min(0).sum()),
                        "top64_mass": float(signed[indices].sum()),
                        "top64_columns": indices.tolist(),
                        "top64_mean_contribution_by_position": means[:, indices].sum(1).tolist(),
                        "selective_count": int((selective > 0).sum()),
                        "selective_top64_columns": selected.tolist(),
                        "selective_top64_mean_contribution_by_position": means[:, selected].sum(1).tolist(),
                    }
                out[name] = {
                    "group": group,
                    "d_style_cosine_by_position": [cosine(x, v) for x in d],
                    "d_style_projection_by_position": (d @ unit).tolist(),
                    "mlp_net_contribution_by_position": means.sum(1).tolist(),
                    "mean_column_cosine": float((projected / down.norm(dim=0).clamp_min(1e-12)).mean()),
                    "column_cosine_quantiles": torch.quantile(projected / down.norm(dim=0).clamp_min(1e-12), torch.tensor([0., .01, .5, .99, 1.])).tolist(),
                    **directions,
                }
            result["layers"][layer] = out
    path = ROOT / "runs/writer_diagnostic.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path = ROOT / "results/writer_summary.json"
    summary_path.write_text(json.dumps(compact_summary(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path)
    print(summary_path)
    for layer in [7, 11, 13, 15, 17, 19, 21, 23]:
        line = [f"L{layer}"]
        for name in ["noun_nanoda_desu", "other_noda_period"]:
            x = result["layers"][layer][name]
            line.append(f"{name}: cos={x['d_style_cosine_by_position'][5]:.3f}, dproj={x['d_style_projection_by_position'][5]:.3f}, net={x['mlp_net_contribution_by_position'][5]:.3f}, plus/minus64={x['promoters']['top64_mass']:.3f}/{x['suppressors']['top64_mass']:.3f}, selective+/-={x['promoters']['selective_count']}/{x['suppressors']['selective_count']}")
        print(" | ".join(line))


if __name__ == "__main__":
    main()
