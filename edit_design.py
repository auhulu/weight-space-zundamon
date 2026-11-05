"""Four predeclared, train-activation-only rank-one edit hypotheses.

No loss, gradients, optimizer, iterative fitting, or strength sweep is used.
Each candidate must be applied to an independently restored base model.
"""
import math

import torch


def _vector(value):
    return torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float32)


def _measure(row):
    norm = float(_vector(row["d_mean"]).norm())
    rms = float(row["d_rms"])
    residual = float(row["residual_rms"])
    coherence = min(1.0, norm / max(rms, 1e-12))
    relative = norm / max(residual, 1e-12)
    return coherence * relative, coherence, relative


def design_candidates(stats):
    """Select layer locations from TRAIN statistics, never evaluation scores."""
    rows = sorted(stats["layers"], key=lambda row: int(row["layer"]))
    if not rows:
        raise ValueError("No train layer statistics")
    for row in rows:
        for key in ("h_mean", "d_mean", "h_second_moment"):
            if not torch.isfinite(_vector(row[key])).all():
                raise ValueError(f"Nonfinite {key}")
        if not all(math.isfinite(float(row[k])) and float(row[k]) > 0
                   for k in ("d_rms", "residual_rms")):
            raise ValueError("Invalid activation scale")
    # Exclude the earliest quarter: their contrasts primarily encode instruction
    # and position changes. Middle and late are distinct structural hypotheses.
    n = len(rows)
    middle = rows[n // 4:max(n // 4 + 1, 3 * n // 4)]
    late = rows[3 * n // 4:]
    eligible = rows[n // 4:]
    ranked = lambda pool: sorted(pool, key=lambda r: (-_measure(r)[0], int(r["layer"])))
    mid, end, best = ranked(middle)[0], ranked(late)[0], ranked(eligible)[0]
    top = ranked(eligible)[:3]

    def spec(name, selected, key_mode, rationale):
        return {"id": name, "method": "rank_one", "key_mode": key_mode,
                "residual_fraction_cap": 0.10, "rationale": rationale,
                "edits": [{"layer": int(r["layer"]), "module_name": r["module_name"],
                           "share": 1.0 / len(selected),
                           "train_score": _measure(r)[0],
                           "train_coherence": _measure(r)[1]} for r in selected]}

    return [
        spec("middle_mean", [mid], "mean", "Most coherent relative contrast in middle half"),
        spec("late_mean", [end], "mean", "Most coherent relative contrast in final quarter"),
        spec("distributed_mean", top, "mean", "Split one contrast budget over three strongest eligible layers"),
        spec("diagonal_key", [best], "diagonal", "Diagonal second-moment preconditioning suppresses high-variance key coordinates"),
    ]


@torch.no_grad()
def apply_edit(model, stats, spec):
    """Mutate only down_proj weights and report actual stored-dtype changes."""
    if spec.get("method") != "rank_one" or spec.get("key_mode") not in ("mean", "diagonal"):
        raise ValueError("Unknown edit method")
    rows = {int(r["layer"]): r for r in stats["layers"]}
    details = []
    for edit in spec["edits"]:
        row = rows[int(edit["layer"])]
        name = edit["module_name"]
        if name != row["module_name"] or not name.endswith(".mlp.down_proj"):
            raise ValueError("Only the specified MLP down projection may be edited")
        weight = model.get_submodule(name).weight
        h, d = _vector(row["h_mean"]), _vector(row["d_mean"])
        if tuple(weight.shape) != (d.numel(), h.numel()):
            raise ValueError("Activation and weight dimensions differ")
        key = h.clone()
        if spec["key_mode"] == "diagonal":
            moment = _vector(row["h_second_moment"])
            floor = max(float(moment.mean()) * 0.10, 1e-12)
            key = h / moment.clamp_min(floor)
        denominator = float(torch.dot(h, key))
        if denominator <= 1e-12:
            raise ValueError("Degenerate mean activation")
        # Fixed safety cap, shared by every hypothesis; no search over strengths.
        cap = float(spec["residual_fraction_cap"]) * float(row["residual_rms"])
        scale = min(1.0, cap / max(float(d.norm()), 1e-12)) * float(edit["share"])
        delta = torch.outer(d * scale, key) / denominator
        before = weight.detach().clone()
        weight.copy_((weight.float() + delta.to(weight.device)).to(weight.dtype))
        realized = weight.float() - before.float()
        details.append({"module_name": name, "layer": int(edit["layer"]),
                        "scale": scale, "changed_elements": int((weight != before).sum().item()),
                        "total_elements": weight.numel(),
                        "relative_norm": float(realized.norm().item() / max(before.float().norm().item(), 1e-12)),
                        "requested_delta_norm": float(delta.norm()),
                        "realized_delta_norm": float(realized.norm().item())})
    return {"id": spec["id"], "changed_elements": sum(x["changed_elements"] for x in details),
            "changed_names": [x["module_name"] for x in details if x["changed_elements"]], "edits": details}
