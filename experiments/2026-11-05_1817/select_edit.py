"""Freeze a reviewed candidate before looking at edited evaluation responses."""
import argparse
import hashlib
import json
import experiment as e

def main(candidate):
    if (e.ROOT/'results/edited_questions_128.json').exists():
        raise RuntimeError('Final evaluation already exists; do not reselect on its outputs')
    review_path=e.ROOT/f'reviews/validation_{candidate}.json'
    reviews=json.loads(review_path.read_text(encoding='utf-8'))
    expected={r['id'] for r in e.read('validation')}
    assert len(reviews)==len(expected) and {r['id'] for r in reviews}==expected
    assert all(type(r['style_ok']) is bool and r['reason'] for r in reviews)
    specs=[r for path in (e.ROOT/'results').glob('*_search.json')
           for r in json.loads(path.read_text(encoding='utf-8'))['finalists'] if r['id']==candidate]
    assert len(specs)==1
    delta_path=e.ROOT/f'models/{candidate}_delta.safetensors'
    with delta_path.open('rb') as f: delta_hash=hashlib.file_digest(f,'sha256').hexdigest()
    record={'id':candidate,'spec':specs[0]['spec'],'validation_n':len(reviews),
        'validation_style_success':sum(r['style_ok'] for r in reviews),
        'validation_style_rate':sum(r['style_ok'] for r in reviews)/len(reviews),
        'criterion':'Natural target ending only; content correctness excluded',
        'selected_before_edited_100_and_generalization':True,
        'reviewer':'Codex agent, not independent human assessment',
        'delta_sha256':delta_hash,'review_sha256':hashlib.sha256(review_path.read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        'text_hash_policy':'UTF-8 with LF line endings'}
    e.dump(e.ROOT/'results/selection.json',record)
    files=[*e.ROOT.glob('*.py'),e.ROOT/'config.json',* (e.ROOT/'data').glob('*.json')]
    e.dump(e.ROOT/'results/frozen_manifest.json',{
        'scope':'Final source/data snapshot at selection; not an initial pre-registration',
        'text_hash_policy':'UTF-8 with LF line endings',
        'sha256':{str(p.relative_to(e.ROOT)):hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in sorted(files)}})
    print(json.dumps(record,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('candidate'); main(p.parse_args().candidate)
