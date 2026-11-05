"""Combine generated responses with explicit per-answer content/style reviews."""

import argparse
import json
from pathlib import Path

from scoring import score, summarize


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def aggregate(rows):
    n = len(rows)
    complete = bool(n) and all(r['review'] is not None for r in rows)
    content_complete = complete and all('content_ok' in r['review'] for r in rows)
    return {
        **summarize(rows),
        'reviewed': sum(r['review'] is not None for r in rows),
        'review_complete': complete,
        'style_rate': sum(r['review']['style_ok'] for r in rows) / n if complete else None,
        'content_rate': sum(r['review']['content_ok'] for r in rows) / n if content_complete else None,
        'joint_rate': sum(r['review']['style_ok'] and r['review']['content_ok'] for r in rows) / n if content_complete else None,
    }


def evaluate(log, reviews=None):
    responses = log['responses']
    ids = [r['id'] for r in responses]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate response IDs')
    indexed = {}
    for review in reviews or []:
        rid = review['id']
        if rid in indexed or rid not in ids:
            raise ValueError(f'Duplicate or unknown review ID: {rid}')
        if type(review['style_ok']) is not bool or ('content_ok' in review and type(review['content_ok']) is not bool):
            raise ValueError(f'Review flags must be booleans: {rid}')
        if not isinstance(review.get('reason'), str) or not review['reason'].strip():
            raise ValueError(f'Review reason required: {rid}')
        indexed[rid] = review
    rows = [{**r, 'heuristic': score(r['answer']), 'review': indexed.get(r['id'])} for r in responses]
    return {
        'generation': {k: v for k, v in log.items() if k != 'responses'},
        'metrics': aggregate(rows),
        'by_category': {c: aggregate([r for r in rows if r['category'] == c]) for c in sorted({r['category'] for r in rows})},
        'responses': rows,
    }


def compare(conditions, baseline='base', edited='edited'):
    if baseline not in conditions or edited not in conditions:
        return {'assessable': False, 'goal_met': None, 'reason': 'Baseline or edited condition missing'}
    a, b = conditions[baseline], conditions[edited]
    paired = [(r['id'], r['prompt']) for r in a['responses']] == [(r['id'], r['prompt']) for r in b['responses']]
    settings = ['system', 'max_new_tokens', 'enable_thinking', 'decoding']
    same_settings = all(k in a['generation'] and k in b['generation'] and a['generation'][k] == b['generation'][k] for k in settings)
    if not paired or not same_settings:
        return {'assessable': False, 'goal_met': None, 'reason': 'Questions or generation settings differ'}
    am, bm = a['metrics'], b['metrics']
    if not (am['review_complete'] and bm['review_complete']):
        return {'assessable': False, 'goal_met': None, 'reason': 'Complete per-answer reviews required; lexical screening alone is insufficient'}
    drop = am['content_rate'] - bm['content_rate'] if am['content_rate'] is not None and bm['content_rate'] is not None else None
    return {'assessable': True, 'style_target': 0.8,
            'success_criterion': 'Reviewed target-ending rate only; content quality is not an acceptance criterion.',
            'content_drop_reference_only': drop, 'goal_met': bm['style_rate'] >= 0.8}


def mapping(values):
    out = {}
    for value in values:
        name, sep, path = value.partition('=')
        if not sep or not name or not path or name in out:
            raise ValueError('Expected unique NAME=PATH arguments')
        out[name] = path
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', action='append', required=True, metavar='NAME=LOG_JSON')
    parser.add_argument('--review', action='append', default=[], metavar='NAME=REVIEW_JSON')
    parser.add_argument('--baseline', default='base')
    parser.add_argument('--edited', default='edited')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    logs, reviews = mapping(args.condition), mapping(args.review)
    if set(reviews) - set(logs):
        parser.error('A review was supplied for an unknown condition')
    conditions = {name: evaluate(read(path), read(reviews[name]) if name in reviews else None) for name, path in logs.items()}
    result = {'review_notice': 'Style is reviewed by an agent. Optional content scores are reference only and do not affect success.',
              'conditions': conditions, 'comparison': compare(conditions, args.baseline, args.edited)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'metrics': {k: v['metrics'] for k, v in conditions.items()}, 'comparison': result['comparison']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
