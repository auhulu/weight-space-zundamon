import pytest

from report import compare, evaluate
from scoring import score


@pytest.mark.parametrize('text', ['お茶を飲むのだ。', 'これは大切なのだ！', 'ぼくも一緒に考えるのだ'])
def test_style_candidate(text):
    assert score(text)['style_candidate']


@pytest.mark.parametrize('text', ['', 'のだ。', 'なのだ！', '「お茶を飲むのだ」と言いました。', '```お茶を飲むのだ。```', '楽しいですのだ。', '行きますなのだ。'])
def test_invalid_style(text):
    assert not score(text)['style_candidate']


def test_repetition_is_reference_only():
    result = score('お茶を飲むのだ。' * 3)
    assert result['repetition']
    assert result['style_candidate']


@pytest.mark.parametrize('text', ['答えは四です。\nのだ', '答えは四です。なのだ。', '答えは四です。\n**のだ**'])
def test_standalone_suffix_not_counted(text):
    result = score(text)
    assert result['target_endings'] == 0
    assert not result['style_candidate']


def test_standalone_suffix_does_not_inflate_valid_count():
    result = score('答えは四なのだ。\nのだ。\nなのだ。')
    assert result['target_endings'] == 1
    assert result['style_candidate']


def log(answer='お茶を飲むのだ。'):
    return {'system': '一般指示', 'max_new_tokens': 128, 'enable_thinking': False, 'decoding': 'greedy',
            'responses': [{'id': 'q1', 'category': 'daily', 'prompt': '休憩には？', 'answer': answer, 'hit_limit': False}]}


def review(content=True, style=True):
    return [{'id': 'q1', 'content_ok': content, 'style_ok': style, 'reason': '質問に対応した自然な回答。'}]


def test_no_semantic_claim_without_reviews():
    result = evaluate(log())
    assert result['metrics']['content_rate'] is None
    assert result['metrics']['joint_rate'] is None
    assert compare({'base': result, 'edited': result})['goal_met'] is None


def test_complete_reviews_style_only_success():
    base = evaluate(log(), review(style=False))
    good = evaluate(log(), review())
    bad = evaluate(log(), review(content=False))
    assert compare({'base': base, 'edited': good})['goal_met'] is True
    assert compare({'base': base, 'edited': bad})['goal_met'] is True


def test_content_review_optional():
    style_review = [{'id': 'q1', 'style_ok': True, 'reason': '自然な目標語尾。'}]
    result = evaluate(log(), style_review)
    assert result['metrics']['review_complete']
    assert result['metrics']['content_rate'] is None
    assert compare({'base': result, 'edited': result})['goal_met'] is True


def test_mismatched_settings_not_comparable():
    base_log = log()
    base_log['max_new_tokens'] = 64
    assert compare({'base': evaluate(base_log, review()), 'edited': evaluate(log(), review())})['goal_met'] is None


def test_duplicate_review_rejected():
    with pytest.raises(ValueError):
        evaluate(log(), review() * 2)
