"""Pure CPU construction of class-selective MLP rank-three deltas.

The observation records must retain continuation_50/data/pairs.json order:
20 nouns, 20 verbs, then 20 adjectives. No file or model is read or modified.
Only means, Gram-Schmidt projections, and outer products construct the edit.
"""
import math

import torch

CLASSES = ("noun", "verb", "adjective")
BRANCH = 5
# Position 6 duplicates branch in all 60 observations and must not be removed.
NUISANCE = (0, 1, 2, 3, 4, 7, 8)
CLASS_SLICES = (slice(0, 20), slice(20, 40), slice(40, 60))


def orthogonal_key(target, negative):
    """Project away observed nuisance means without solving a fitted system."""
    if not torch.isfinite(target).all() or target.norm() < 1e-8:
        raise ValueError("Nonfinite or degenerate target activity")
    basis = []
    for vector in negative:
        if not torch.isfinite(vector).all():
            raise ValueError("Nonfinite nuisance activity")
        residual = vector.clone()
        # Reorthogonalization controls float32 cancellation for dependent means.
        for _ in range(2):
            for unit in basis:
                residual -= unit * torch.dot(unit, residual)
        if residual.norm() > 1e-7:
            basis.append(residual / residual.norm())
    key = target.clone()
    for _ in range(2):
        for unit in basis:
            key -= unit * torch.dot(unit, key)
    if not torch.isfinite(key).all() or key.norm() < target.norm() * .01:
        raise ValueError("Degenerate class key after nuisance projection")
    return key


@torch.no_grad()
def make_edit(stats, spec):
    """Return ``(parameter_name -> delta, layer -> diagnostics)``.

    Required spec: layers and alpha. class_nuisance defaults to False.
    False removes 7 global position means; True removes all 21 class-position
    means. Both remove the other two class branch means for each class key.
    Alpha is divided across layers, never across the three classes.
    """
    layers = spec["layers"]
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("layers must be nonempty and unique")
    alpha = float(spec["alpha"])
    if not math.isfinite(alpha):
        raise ValueError("alpha must be finite")
    class_nuisance = spec.get("class_nuisance", False)
    if not isinstance(class_nuisance, bool):
        raise ValueError("class_nuisance must be a boolean")
    edits, details = {}, {}
    for layer in layers:
        observed = stats[layer]
        g = observed["g"].detach().to(device="cpu", dtype=torch.float32)
        d = observed["d_style"].detach().to(device="cpu", dtype=torch.float32)
        if (g.ndim != 3 or d.ndim != 3 or g.shape[:2] != d.shape[:2]
                or g.shape[0] != 60 or g.shape[1] != 9
                or g.shape[2] == 0 or d.shape[2] == 0):
            raise ValueError("Expected 60 records and 9 positions in g/d_style")
        if not torch.isfinite(g).all() or not torch.isfinite(d).all():
            raise ValueError("Nonfinite observed activity or direction")
        class_means = torch.stack([g[part, BRANCH].mean(0) for part in CLASS_SLICES])
        directions = torch.stack([d[part, BRANCH].mean(0) for part in CLASS_SLICES])
        if class_nuisance:
            nuisance = [g[part, pos].mean(0)
                        for part in CLASS_SLICES for pos in NUISANCE]
        else:
            nuisance = [g[:, pos].mean(0) for pos in NUISANCE]
        keys, fractions = [], []
        for index, target in enumerate(class_means):
            negative = nuisance + [class_means[j] for j in range(3) if j != index]
            key = orthogonal_key(target, negative)
            denominator = torch.dot(target, key)
            if not torch.isfinite(denominator) or denominator <= 0:
                raise ValueError("Invalid class-key normalization")
            fractions.append(float(key.norm() / target.norm()))
            keys.append(key / denominator)
        keys = torch.stack(keys)
        # Each class has unit response to its own mean and zero to the others.
        delta = sum(torch.outer(direction, key)
                    for direction, key in zip(directions, keys)) * (alpha / len(layers))
        if not torch.isfinite(delta).all():
            raise ValueError("Nonfinite weight delta")
        gain = g @ keys.T
        quantiles = torch.tensor([.1, .5, .9])
        details[layer] = {
            "method": "class_rank3",
            "class_order": list(CLASSES),
            "class_nuisance": class_nuisance,
            "nuisance_mean_count": len(nuisance),
            "nuisance_positions": list(NUISANCE),
            "alpha_per_class_this_layer": alpha / len(layers),
            "key_fraction": min(fractions),
            "delta_norm": float(delta.norm()),
            "gain_quantiles": torch.quantile(gain.sum(-1), quantiles, dim=0).tolist(),
            "class_keys": {
                name: {
                    "key_fraction": fractions[i],
                    "gain_quantiles": torch.quantile(gain[:, :, i], quantiles, dim=0).tolist(),
                } for i, name in enumerate(CLASSES)
            },
            "branch_cross_mean_gains": (class_means @ keys.T).tolist(),
            "max_abs_nuisance_mean_gain": float((torch.stack(nuisance) @ keys.T).abs().max()),
        }
        edits[f"model.layers.{layer}.mlp.down_proj.weight"] = delta
    return edits, details
