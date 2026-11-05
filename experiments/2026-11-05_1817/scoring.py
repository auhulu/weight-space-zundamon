"""Lexical screening only: naturalness and content require a separate review."""

import re
from collections import Counter


def remove_quotes(text):
    pairs = {'「': '」', '『': '』', '“': '”', '"': '"', '`': '`'}
    stack, out = [], []
    for char in text:
        if stack and char == stack[-1]:
            stack.pop()
        elif char in pairs:
            stack.append(pairs[char])
        elif not stack:
            out.append(char)
    return ''.join(out)


def score(text):
    """Return mechanical indicators, never a claim of semantic correctness."""
    unfenced = re.sub(r'```[\s\S]*?(?:```|$)|~~~[\s\S]*?(?:~~~|$)', '', text)
    clean = remove_quotes(unfenced).strip()
    sentences = [s.strip() for s in re.split(r'[。！？!?\n]+', clean) if s.strip()]
    target = re.compile(r'(?:なのだ|のだ)[\s\u200b]*$')
    malformed = re.compile(r'(?:です|ます|でした|ました|ません|だ)(?:なのだ|のだ)[\s\u200b]*$')
    def has_body(sentence):
        remainder = re.sub(r'(?:なのだ|のだ)|[\s\u200b。！？!?、,.…〜~ー*#\-]', '', sentence)
        return bool(re.search(r'\w', remainder))
    endings = sum(bool(target.search(s)) and not malformed.search(s) and has_body(s) for s in sentences)
    normalized = re.sub(r'\s+', '', clean)
    remainder = re.sub(r'(?:なのだ|のだ)|[。！？!?、,.…〜~ー*#\-]', '', normalized)
    style_only = bool(normalized) and not remainder
    empty = not bool(re.search(r'\w', text))
    repeated_sentence = any(n >= 3 for n in Counter(sentences).values())
    repeated_fragment = bool(re.search(r'(.{2,30})\1{3,}', normalized))
    repetition = repeated_sentence or repeated_fragment
    return {
        'sentences': len(sentences),
        'target_endings': endings,
        'ending_rate': endings / len(sentences) if sentences else 0.0,
        'target_mention': bool(re.search(r'(?:なのだ|のだ)', clean)),
        'empty': empty,
        'style_only': style_only,
        'repetition': repetition,
        'malformed_ending': any(malformed.search(s) is not None for s in sentences),
        'style_candidate': bool(endings and not (empty or style_only)),
    }


def summarize(responses):
    scored = [score(r['answer']) for r in responses]
    n = len(responses)
    return {
        'n': n,
        'heuristic_style_rate': sum(s['style_candidate'] for s in scored) / n if n else 0.0,
        'empty': sum(s['empty'] for s in scored),
        'style_only': sum(s['style_only'] for s in scored),
        'repetition': sum(s['repetition'] for s in scored),
        'malformed_ending': sum(s['malformed_ending'] for s in scored),
        'hit_limit': sum(bool(r.get('hit_limit', False)) for r in responses),
    }
