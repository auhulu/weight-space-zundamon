"""CPU-only checks of class-selective geometric weight construction."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent
module_spec = importlib.util.spec_from_file_location("class_edit", ROOT / "class_edit.py")
edit = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(edit)


def observations(layers=(19,)):
    g = torch.zeros(60, 9, 40)
    d = torch.zeros(60, 9, 5)
    for ci, part in enumerate(edit.CLASS_SLICES):
        for j, pos in enumerate(edit.NUISANCE):
            # Class-specific nuisance means differ; their global mean does not
            # span the same space as all 21 individual class-position means.
            g[part, pos, ci * 7 + j] = 1
            g[part, pos, 24 + j] = .25
        g[part, 5, 21 + ci] = 1
        g[part, 5] += .3 * g[part][:, list(edit.NUISANCE)].sum(1)
        g[part, 5, 35 + ci] += torch.linspace(-.05, .05, 20)
        g[part, 6] = g[part, 5]
        d[part, 5, ci] = 1
        d[part, 5, 3] = .1 * (ci + 1)
    return {layer: {"g": g.clone(), "d_style": d.clone()} for layer in layers}


@pytest.mark.parametrize("class_nuisance", [False, True])
def test_mean_class_responses_are_independent_and_write_their_own_direction(class_nuisance):
    stats = observations()
    deltas, details = edit.make_edit(stats, {
        "layers": [19], "alpha": 1.25, "class_nuisance": class_nuisance,
    })
    delta = deltas["model.layers.19.mlp.down_proj.weight"]
    assert delta.device.type == "cpu"
    assert torch.allclose(torch.tensor(details[19]["branch_cross_mean_gains"]),
                          torch.eye(3), atol=2e-6)
    for part in edit.CLASS_SLICES:
        target = stats[19]["d_style"][part, 5].mean(0)
        assert torch.allclose(delta @ stats[19]["g"][part, 5].mean(0),
                              1.25 * target, atol=2e-6)


@pytest.mark.parametrize("class_nuisance, count", [(False, 7), (True, 21)])
def test_requested_nuisance_means_receive_no_edit(class_nuisance, count):
    stats = observations()
    deltas, details = edit.make_edit(stats, {
        "layers": [19], "alpha": 1., "class_nuisance": class_nuisance,
    })
    g = stats[19]["g"]
    nuisance = ([g[part, pos].mean(0) for part in edit.CLASS_SLICES for pos in edit.NUISANCE]
                if class_nuisance else [g[:, pos].mean(0) for pos in edit.NUISANCE])
    delta = deltas["model.layers.19.mlp.down_proj.weight"]
    assert details[19]["nuisance_mean_count"] == count
    assert all(torch.allclose(delta @ vector, torch.zeros(5), atol=2e-6)
               for vector in nuisance)


def test_layer_strength_is_split_and_input_stats_are_unchanged():
    stats = observations((19, 20))
    before = {layer: {k: v.clone() for k, v in values.items()} for layer, values in stats.items()}
    spec = {"layers": [19, 20], "alpha": 2., "class_nuisance": True}
    first, _ = edit.make_edit(stats, spec)
    second, _ = edit.make_edit(stats, spec)
    for layer in [19, 20]:
        key = f"model.layers.{layer}.mlp.down_proj.weight"
        assert torch.equal(first[key], second[key])
        for part in edit.CLASS_SLICES:
            assert torch.allclose(first[key] @ stats[layer]["g"][part, 5].mean(0),
                                  stats[layer]["d_style"][part, 5].mean(0), atol=2e-6)
        assert all(torch.equal(stats[layer][k], value) for k, value in before[layer].items())


@pytest.mark.parametrize("field", ["g", "d_style"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_observations_are_rejected(field, bad):
    stats = observations()
    stats[19][field][0, 5, 0] = bad
    with pytest.raises(ValueError, match="Nonfinite"):
        edit.make_edit(stats, {"layers": [19], "alpha": 1.})


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_strength_is_rejected(bad):
    with pytest.raises(ValueError, match="alpha"):
        edit.make_edit(observations(), {"layers": [19], "alpha": bad})


@pytest.mark.parametrize("remaining", [0., 1e-5])
def test_indistinguishable_class_keys_are_rejected(remaining):
    stats = observations()
    stats[19]["g"][:20, 5] = stats[19]["g"][20:40, 5].mean(0)
    stats[19]["g"][:20, 5, 39] += remaining
    with pytest.raises(ValueError, match="Degenerate"):
        edit.make_edit(stats, {"layers": [19], "alpha": 1.})


def test_default_class_slices_match_saved_observation_order():
    pairs = json.loads((ROOT.parent / "continuation_50/data/pairs.json").read_text(encoding="utf-8"))
    assert len(pairs) == 60
    for name, part in zip(edit.CLASSES, edit.CLASS_SLICES):
        assert all(row["category"].startswith(name + "_") for row in pairs[part])
