"""CPU checks for geometric intervention and observation/evaluation separation."""
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent
module_spec = importlib.util.spec_from_file_location('continuation50_run', ROOT / 'run.py')
run = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(run)


def stats_fixture(layers=(7,)):
    g = torch.zeros(4, 9, 6)
    g[:, 0] = torch.tensor([1., 0., 0., 0., 0., 0.])
    # The second nuisance mean is deliberately not orthogonal to the first.
    g[:, 1] = torch.tensor([1., 2., 0., 0., 0., 0.])
    g[:, 2] = torch.tensor([1., 2., 3., 4., 5., 6.])
    g[:, 2, 5] += torch.tensor([-.2, -.1, .1, .2])
    d = torch.zeros(4, 9, 3)
    d[:, 2] = torch.tensor([.2, -.3, .5])
    return {layer: {'g': g.clone(), 'd_style': d.clone(), 'd_total': 2*d.clone()}
            for layer in layers}


def test_orthogonal_key_removes_nonorthogonal_nuisance_means():
    g = torch.tensor([1., 2., 3., 4.])
    negatives = [torch.tensor([1., 0., 0., 0.]), torch.tensor([1., 2., 0., 0.])]
    key = run.orthogonal_key(g, negatives)
    assert torch.isfinite(key).all()
    assert torch.allclose(key, torch.tensor([0., 0., 3., 4.]), atol=1e-6)
    assert all(abs(float(key @ v)) < 1e-6 for v in negatives)
    assert torch.allclose(g @ key, key @ key, atol=1e-6)


def test_unmasked_mean_key_writes_target_direction():
    stats = stats_fixture()
    edits, details = run.make_edit(stats, {'layers':[7], 'direction':'d_style',
                                         'alpha':2.5, 'negative':[]})
    delta = edits['model.layers.7.mlp.down_proj.weight']
    g = stats[7]['g'][:, 2].mean(0)
    d = stats[7]['d_style'][:, 2].mean(0)
    assert delta.shape == (3, 6)
    assert torch.allclose(delta @ g, 2.5*d, atol=1e-6)
    assert torch.isfinite(delta).all()
    assert details[7]['key_fraction'] == pytest.approx(1.)


def test_projected_edit_writes_target_and_zero_to_nuisance_means():
    stats = stats_fixture()
    edits, _ = run.make_edit(stats, {'layers':[7], 'direction':'d_total',
                                    'alpha':3., 'negative':[0,1]})
    delta = edits['model.layers.7.mlp.down_proj.weight']
    g = stats[7]['g']
    d = stats[7]['d_total'][:, 2].mean(0)
    assert torch.allclose(delta @ g[:,2].mean(0), 3*d, atol=1e-6)
    for position in [0,1]:
        assert torch.allclose(delta @ g[:,position].mean(0), torch.zeros(3), atol=1e-6)


def test_multilayer_edit_divides_strength_without_changing_stats():
    stats = stats_fixture((7,11))
    original_g = stats[7]['g'].clone()
    edits, _ = run.make_edit(stats, {'layers':[7,11], 'direction':'d_style',
                                    'alpha':4., 'negative':[0,1]})
    for layer in [7,11]:
        delta = edits[f'model.layers.{layer}.mlp.down_proj.weight']
        assert torch.allclose(delta @ stats[layer]['g'][:,2].mean(0),
                              2*stats[layer]['d_style'][:,2].mean(0), atol=1e-6)
    assert torch.equal(stats[7]['g'], original_g)


@pytest.mark.parametrize('g,negative', [
    (torch.zeros(3), []),
    (torch.tensor([1.,2.,0.]), [torch.tensor([1.,0.,0.]),torch.tensor([1.,2.,0.])]),
    (torch.tensor([1.,0.,1e-5]), [torch.tensor([1.,0.,0.])]),
])
def test_zero_and_degenerate_keys_rejected(g, negative):
    with pytest.raises(ValueError):
        run.orthogonal_key(g, negative)


def test_nonfinite_edit_is_rejected():
    stats = stats_fixture()
    with pytest.raises((AssertionError, ValueError)):
        run.make_edit(stats, {'layers':[7], 'direction':'d_style',
                             'alpha':float('inf'), 'negative':[]})


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nonfinite_target_key_is_rejected(bad):
    with pytest.raises(ValueError):
        run.orthogonal_key(torch.tensor([1.,bad,3.]), [])


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nonfinite_nuisance_key_is_rejected(bad):
    with pytest.raises(ValueError):
        run.orthogonal_key(torch.tensor([1.,2.,3.]), [torch.tensor([bad,0.,0.])])


def test_observation_and_evaluation_questions_are_disjoint():
    def read(path): return json.loads(path.read_text(encoding='utf-8'))
    evaluation = read(ROOT.parent/'data/questions.json')
    pairs = read(ROOT/'data/pairs.json')
    heldout = read(ROOT/'data/heldout.json')
    assert [len(x) for x in [evaluation,pairs,heldout]] == [100,60,50]
    prompts = [r['prompt'] for group in [evaluation,pairs,heldout] for r in group]
    assert len(prompts) == len(set(prompts))
    assert Counter(r['category'].split('_')[0] for r in pairs) == {'noun':20,'verb':20,'adjective':20}
    assert set(Counter(r['category'] for r in heldout).values()) == {5}
    for row in pairs:
        expected = row['normal'][:-2]+'なのだ。' if row['category'].startswith('noun') else row['normal'][:-1]+'のだ。'
        assert row['style'] == expected
