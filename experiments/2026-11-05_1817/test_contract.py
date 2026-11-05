"""CPU-only tests of edit invariants and prompt/generation contracts."""
from types import SimpleNamespace

import pytest
import torch

import experiment as e

NAME = 'model.layers.0.mlp.down_proj.weight'


class ToyModel:
    def __init__(self):
        self.weight = torch.nn.Parameter(torch.arange(6).reshape(2, 3).float(), requires_grad=False)

    def get_parameter(self, name):
        assert name == NAME
        return self.weight


def test_non_cumulative_and_restore():
    model = ToyModel()
    original = model.weight.detach().clone()
    e.apply(model, {NAME: original}, {NAME: torch.ones_like(original)})
    assert torch.equal(model.weight, original + 1)
    e.apply(model, {NAME: original}, {NAME: torch.full_like(original, 2)})
    assert torch.equal(model.weight, original + 2)
    e.apply(model, {NAME: original}, {})
    assert torch.equal(model.weight, original)


@pytest.mark.parametrize('name', ['lm_head.weight', 'model.embed_tokens.weight', 'model.norm.weight', 'model.layers.0.input_layernorm.weight'])
def test_prohibited_targets(name):
    model = ToyModel()
    original = model.weight.detach().clone()
    with pytest.raises(ValueError, match='Only internal'):
        e.apply(model, {NAME: original}, {name: torch.zeros_like(original)})
    assert torch.equal(model.weight, original)


@pytest.mark.parametrize('invalid', [float('inf'), float('nan')])
def test_nonfinite_rejected_before_mutation(invalid):
    model = ToyModel()
    original = model.weight.detach().clone()
    with pytest.raises(ValueError, match='Nonfinite'):
        e.apply(model, {NAME: original}, {NAME: torch.full_like(original, invalid)})
    assert torch.equal(model.weight, original)


def test_shape_mismatch_rejected():
    model = ToyModel()
    with pytest.raises(ValueError, match='shape'):
        e.apply(model, {NAME: model.weight.detach().clone()}, {NAME: torch.ones(3, 2)})


def stats():
    return {0: {'g': torch.tensor([[1., 2., 3.], [2., 3., 4.]]),
                'd': torch.tensor([[1., 2.], [2., 3.]]),
                'u': torch.tensor([1., 0.]), 'writer': torch.tensor([1., 3., 2.])}}


@pytest.mark.parametrize('method', ['write', 'project'])
def test_sparse_edit_shape_and_untouched_columns(method):
    original = torch.arange(6).reshape(2, 3).float()
    edit = e.mlp_delta(stats(), {0: original}, {'layers': [0], 'columns': 1, 'alpha': 2., 'method': method})[NAME]
    assert edit.shape == original.shape
    assert torch.equal(edit[:, [0, 2]], torch.zeros(2, 2))
    assert torch.isfinite(edit).all()
    assert edit[:, 1].abs().sum() > 0


def test_regression_method_disallowed():
    with pytest.raises(ValueError, match='Only hypothesis-driven'):
        e.mlp_delta(stats(), {0: torch.ones(2, 3)}, {'layers': [0], 'columns': 1, 'alpha': 1., 'method': 'ridge'})


class Batch(dict):
    def to(self, device):
        return self


class Tokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        self.messages, self.kwargs = messages, kwargs
        return Batch(input_ids=torch.tensor([[10, 11]] * len(messages)), attention_mask=torch.ones(len(messages), 2, dtype=torch.long))

    def decode(self, tokens, **kwargs):
        return '回答なのだ。'


def test_unsteered_prompt_and_disabled_thinking():
    tok = Tokenizer()
    e.inputs(tok, ['質問一', '質問二'])
    assert e.CFG['max_new_tokens'] == 128
    assert e.CFG['system'] == '日本語で簡潔に、原則1文で答えてください。'
    assert tok.messages == [[{'role': 'system', 'content': e.CFG['system']}, {'role': 'user', 'content': q}] for q in ['質問一', '質問二']]
    assert tok.kwargs['enable_thinking'] is False
    assert tok.kwargs['add_generation_prompt'] is True


def test_generation_limit_greedy_and_eos_accounting():
    class Model:
        generation_config = SimpleNamespace(eos_token_id=[2, 3])

        def generate(self, **kwargs):
            assert kwargs['max_new_tokens'] == 128
            assert kwargs['do_sample'] is False
            first = [7, 3] + [0] * 126
            return torch.cat([kwargs['input_ids'], torch.tensor([first, [7] * 128])], 1)

    result = e.generate(Model(), Tokenizer(), [{'id': 'a', 'prompt': '一'}, {'id': 'b', 'prompt': '二'}])
    assert result['responses'][0]['tokens'] == 2
    assert result['responses'][0]['hit_limit'] is False
    assert result['responses'][1]['tokens'] == 128
    assert result['responses'][1]['hit_limit'] is True


def test_fixed_revision_and_dataset_separation():
    assert e.CFG['revision'] == '2fc06364715b967f1860aea9cf38778875588b17'
    groups = [e.read(n) for n in ['questions', 'train_pairs', 'validation', 'generalization']]
    assert list(map(len, groups)) == [100, 100, 50, 50]
    all_prompts = [r['prompt'] for group in groups for r in group]
    assert len(all_prompts) == len(set(all_prompts))


def test_standalone_reload_uses_saved_tokenizer(monkeypatch):
    loaded = []
    class Model:
        def to(self, device): return self
        def eval(self): return self
        def requires_grad_(self, value): assert value is False
    monkeypatch.setattr(e.Qwen3_5ForCausalLM, 'from_pretrained', lambda source, **kw: Model())
    def tokenizer(source, **kw):
        loaded.append((source, kw))
        return SimpleNamespace(pad_token_id=0)
    monkeypatch.setattr(e.AutoTokenizer, 'from_pretrained', tokenizer)
    e.load_model('models/zundamon-edited')
    assert loaded == [('models/zundamon-edited', {'local_files_only': True})]
