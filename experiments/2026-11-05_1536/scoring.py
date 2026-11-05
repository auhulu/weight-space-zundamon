"""Predeclared lexical metrics; these do not establish semantic quality."""
import json
import re
from collections import Counter


def remove_quotes(text):
    # Remove quoted content, including nested Japanese quotes, without evaluating it.
    pairs = {'「': '」', '『': '』', '“': '”', '"': '"'}
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
    unfenced = re.sub(r'```[\s\S]*?(?:```|$)|~~~[\s\S]*?(?:~~~|$)', '', text)
    clean = remove_quotes(unfenced).strip()
    sentences = [s.strip() for s in re.split(r'[。！？!?\n]+', clean) if s.strip()]
    endings = sum(bool(re.search(r'(?:なのだ|のだ)\s*$', s)) for s in sentences)
    normalized = re.sub(r'\s+', '', clean)
    style_only = bool(normalized) and not re.sub(r'(?:なのだ|のだ)|[。！？!?、,.…〜~ー]', '', normalized)
    counts = Counter(sentences)
    repeated_sentence = any(n >= 3 for n in counts.values())
    repeated_fragment = bool(re.search(r'(.{2,30})\1{3,}', normalized))
    return {
        'sentences': len(sentences), 'target_endings': endings,
        'ending_rate': endings / len(sentences) if sentences else 0.0,
        'target_mention': bool(re.search(r'(?:なのだ|のだ)', clean)),
        'empty': not bool(re.search(r'\w', text)), 'style_only': style_only,
        'repetition': repeated_sentence or repeated_fragment,
    }


def validate_side(text, validator):
    text = text.strip()
    kind = validator['kind']
    if kind == 'exact':
        return text == validator['value']
    if kind == 'json_equal':
        try:
            return json.loads(text) == validator['value']
        except (ValueError, TypeError):
            return False
    if kind == 'polite':
        sentences = [s.strip() for s in re.split(r'[。！？!?\n]+', remove_quotes(text)) if s.strip()]
        return bool(sentences) and all(re.search(r'(です|ます|でした|ました|ません|ませんでした)$', s) for s in sentences)
    if kind == 'lines_exact':
        return text.splitlines() == validator['value']
    raise ValueError(f'Unknown side validator: {kind}')
