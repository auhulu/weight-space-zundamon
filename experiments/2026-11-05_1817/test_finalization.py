"""CPU simulations of contrast edits and checkpoint export verification."""
import ast
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

import contrast
import experiment as e
import finalize

NAME = 'model.layers.0.mlp.down_proj.weight'


@pytest.mark.parametrize('method', ['write', 'suppress'])
def test_contrast_edits_only_selected_internal_columns(method):
    stats = {0: {'g': torch.tensor([[[1., 2., 3.]] * 3, [[2., 3., 4.]] * 3]),
                 'd': torch.tensor([[[1., 2.]] * 3, [[2., 3.]] * 3])}}
    base = {0: torch.tensor([[1., -2., 3.], [3., -1., 2.]])}
    edits = contrast.delta_for(stats, base, {'method': method, 'layers': [0], 'position': 2, 'columns': 1, 'alpha': 2})
    assert set(edits) == {NAME}
    delta = edits[NAME]
    assert delta.shape == base[0].shape
    assert torch.isfinite(delta).all()
    assert int((delta.abs().sum(0) != 0).sum()) == 1
    assert torch.linalg.matrix_rank(delta) == 1
    if method == 'suppress':
        # The negative writer is column 1; suppression reverses its negative component.
        assert torch.all(delta[:, 1] > 0)


def test_no_training_or_regression_entrypoints():
    prohibited = {'backward', 'fit', 'lstsq', 'solve', 'pinv', 'pinverse', 'Adam', 'AdamW', 'SGD'}
    for module in [e, contrast, finalize]:
        tree = ast.parse(Path(module.__file__).read_text(encoding='utf-8'))
        calls = [n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, (ast.Attribute, ast.Name))]
        assert not set(calls) & prohibited


def gated_stats(branch):
    mean = torch.tensor([[1., 1., 0., 0.], [2., 1., 0., 0.], branch])
    perturbation = torch.tensor([[.1, -.1, .2, -.2], [.2, .1, -.1, .1], [.1, .2, .3, .4]])
    g = torch.stack([mean + perturbation, mean - perturbation])
    d = torch.tensor([[[0., 0.], [0., 0.], [1., 3.]], [[0., 0.], [0., 0.], [3., 5.]]])
    return {0: {'g': g, 'd': d}}


def test_gated_write_preserves_average_start_and_middle():
    stats = gated_stats([3., 5., 2., -1.])
    base = {0: torch.arange(8).reshape(2, 4).float()}
    spec = {'method': 'gated_write', 'layers': [0], 'columns': 4, 'position': 2, 'alpha': 1.}
    delta = contrast.delta_for(stats, base, spec)[NAME]
    mean_g = stats[0]['g'].mean(0)
    target = stats[0]['d'][:, 2].mean(0)
    assert torch.allclose(delta @ mean_g[0], torch.zeros(2), atol=1e-5)
    assert torch.allclose(delta @ mean_g[1], torch.zeros(2), atol=1e-5)
    assert torch.allclose(delta @ mean_g[2], target, atol=1e-5)
    assert torch.linalg.matrix_rank(delta) == 1


@pytest.mark.parametrize('branch', [[3., 2., 0., 0.], [3., 2., 1e-5, 0.], [0., 0., 0., 0.]])
def test_gated_write_rejects_degenerate_branch(branch):
    spec = {'method': 'gated_write', 'layers': [0], 'columns': 4, 'position': 2, 'alpha': 1.}
    with pytest.raises(ValueError, match='Degenerate'):
        contrast.delta_for(gated_stats(branch), {0: torch.ones(2, 4)}, spec)


def test_terminal_gated_preserves_start_middle_and_three_post_suffix_means():
    g = torch.tensor([[1., 1., 0., 0., 0., 0.], [2., 1., 0., 0., 0., 0.], [3., 5., 2., -1., 2., 3.]])
    post = torch.tensor([[1., 0., 1., 0., 0., 0.], [0., 1., 2., 1., 0., 0.], [0., 0., 0., 2., 1., 0.]])
    d = torch.tensor([[0., 0.], [0., 0.], [2., 4.]])
    stats = {0: {'g': torch.stack([g + .1, g - .1]), 'd': torch.stack([d, d]),
                 'post': torch.stack([post + .2, post - .2])}}
    spec = {'method': 'terminal_gated', 'layers': [0], 'columns': 6, 'position': 2, 'alpha': 1.}
    delta = contrast.delta_for(stats, {0: torch.ones(2, 6)}, spec)[NAME]
    for nuisance in [*stats[0]['g'].mean(0)[:2], *stats[0]['post'].mean(0)]:
        assert torch.allclose(delta @ nuisance, torch.zeros(2), atol=1e-5)
    assert torch.allclose(delta @ stats[0]['g'].mean(0)[2], d[2], atol=1e-5)
    assert torch.isfinite(delta).all()


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
        self.model.layers[0].mlp = torch.nn.Module()
        self.model.layers[0].mlp.down_proj = torch.nn.Linear(3, 2, bias=False)
        self.model.layers[0].mlp.down_proj.weight.data.copy_(torch.arange(6).reshape(2, 3))
        self.model.embed_tokens = torch.nn.Embedding(4, 2)
        self.model.embed_tokens.weight.data.copy_(torch.arange(8).reshape(4, 2))
        self.model.norm = torch.nn.LayerNorm(2)
        self.lm_head = torch.nn.Linear(2, 4, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight
        self.config = SimpleNamespace(tie_word_embeddings=True)
        self.saved = None

    def save_pretrained(self, folder):
        Path(folder).mkdir(parents=True)
        self.saved = {n: p.clone() for n, p in self.state_dict().items()}


def setup_finalize(monkeypatch, tmp_path, corrupt_reload=False, zero_edit=False, untie_reload=False):
    root = tmp_path / 'repo/experiments/1817'
    root.mkdir(parents=True)
    monkeypatch.setattr(e, 'ROOT', root)
    source = root.parent.parent / 'cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots' / e.CFG['revision'] / 'model.safetensors-00001-of-00001.safetensors'
    # Avoid Windows MAX_PATH while keeping the real checkpoint hash path exercised.
    path_open = Path.open
    def open_path(path, *args, **kwargs):
        if path == source:
            return io.BytesIO(b'fake checkpoint for CPU test')
        return path_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', open_path)
    model = FakeModel()
    tok = SimpleNamespace(save_pretrained=lambda folder: None)

    def load_model(source=None):
        if source is None:
            return model, tok
        reload = FakeModel()
        reload.load_state_dict(model.saved)
        if corrupt_reload:
            reload.model.layers[0].mlp.down_proj.weight.data.add_(1)
        if untie_reload:
            reload.lm_head.weight = torch.nn.Parameter(reload.lm_head.weight.detach().clone())
        return reload, tok

    def generate(m, t, questions, path=None):
        offset = float(m.model.layers[0].mlp.down_proj.weight[0, 0])
        responses = [{**q, 'answer': f'値{offset}なのだ。', 'hit_limit': False} for q in questions]
        result = {**e.CFG, 'responses': responses}
        if path:
            e.dump(path, result)
        return result

    monkeypatch.setattr(e, 'load_model', load_model)
    monkeypatch.setattr(e, 'generate', generate)
    monkeypatch.setattr(e, 'read', lambda name: [{'id': 'a', 'category': 'test', 'prompt': '試験'}])
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda: 'CPU simulation')
    edit = root / 'edit.safetensors'
    save_file({NAME: torch.zeros(2, 3) if zero_edit else torch.full((2, 3), .5)}, str(edit))
    selection = root / 'selection.json'
    import hashlib
    selection.write_text(json.dumps({'id': 'cpu_test', 'edit_file': 'edit.safetensors', 'delta_sha256':hashlib.sha256(edit.read_bytes()).hexdigest()}), encoding='utf-8')
    return root, edit, selection


def test_finalize_verifies_save_reload_restore_and_metadata(monkeypatch, tmp_path):
    root, edit, selection = setup_finalize(monkeypatch, tmp_path)
    finalize.main(edit, selection)
    verification = json.loads((root / 'results/verification.json').read_text(encoding='utf-8'))
    assert verification['changed_elements'] == {NAME: 6}
    for key in ['noncumulative', 'restore_hashes_equal', 'restore_greedy_equal', 'all_other_parameters_unchanged',
                'reload_hashes_equal', 'reload_greedy_equal', 'embedding_output_final_norm_unchanged', 'weight_tying_preserved']:
        assert verification[key]
    assert len(verification['source_checkpoint_sha256']) == 64
    generated = json.loads((root / 'results/edited_questions_128.json').read_text(encoding='utf-8'))
    assert generated['system'] == e.CFG['system']
    assert generated['max_new_tokens'] == 128
    assert generated['enable_thinking'] is False


def test_finalize_rejects_reload_corruption(monkeypatch, tmp_path):
    root, edit, selection = setup_finalize(monkeypatch, tmp_path, corrupt_reload=True)
    with pytest.raises(AssertionError):
        finalize.main(edit, selection)
    assert not (root / 'results/verification.json').exists()


def test_finalize_rejects_zero_edit(monkeypatch, tmp_path):
    root, edit, selection = setup_finalize(monkeypatch, tmp_path, zero_edit=True)
    with pytest.raises(AssertionError, match='zero edit'):
        finalize.main(edit, selection)
    assert not (root / 'results/verification.json').exists()


def test_finalize_rejects_edit_different_from_frozen_selection(monkeypatch, tmp_path):
    root, edit, selection = setup_finalize(monkeypatch, tmp_path)
    record = json.loads(selection.read_text(encoding='utf-8'))
    record['delta_sha256'] = '0' * 64
    selection.write_text(json.dumps(record), encoding='utf-8')
    with pytest.raises(AssertionError, match='frozen selection'):
        finalize.main(edit, selection)
    assert not (root / 'results/verification.json').exists()


def test_finalize_rejects_broken_weight_sharing(monkeypatch, tmp_path):
    root, edit, selection = setup_finalize(monkeypatch, tmp_path, untie_reload=True)
    with pytest.raises(AssertionError):
        finalize.main(edit, selection)
    assert not (root / 'results/verification.json').exists()
