"""Freeze a reviewed geometric edit before generating held-out answers."""
import argparse
from datetime import datetime, timezone
import json
import run as r
import sys
sys.path.insert(0,str(r.ROOT))
from finalize import sha256
from report import evaluate


def main(candidate, family):
    destination = r.ROOT/'results/selection.json'
    if destination.exists(): raise FileExistsError(destination)
    if (r.ROOT/'results/edited_heldout_128.json').exists():
        raise ValueError('Held-out answers already exist: a fresh selection is not allowed')
    records = json.loads((r.ROOT/f'results/{family}_search.json').read_text(encoding='utf-8'))
    record = next(row for row in records if row['id'] == candidate)
    reviewed = {}
    for split in ['validation','questions']:
        log = json.loads((r.ROOT/f'results/{split}_{candidate}.json').read_text(encoding='utf-8'))
        review = json.loads((r.ROOT/f'reviews/{split}_{candidate}.json').read_text(encoding='utf-8'))
        if log.get('mode') != 'base' or log.get('diagnostic_only') is not False:
            raise ValueError('Prompted diagnostics cannot be selected as an edited model')
        if [(x['id'],x['prompt']) for x in log['responses']] != [(x['id'],x['prompt']) for x in r.e.read(split)]:
            raise ValueError('Evaluation questions changed')
        metrics = evaluate(log,review)['metrics']
        if not metrics['review_complete']: raise ValueError('Incomplete review')
        reviewed[split] = metrics
    if reviewed['questions']['style_rate'] < .5:
        raise ValueError('This freeze entry point requires the requested 50% target')
    delta = r.ROOT/f'models/{candidate}_delta.safetensors'
    paths = ['run.py','data/pairs.json','data/heldout.json',
             f'results/{family}_search.json',f'results/questions_{candidate}.json',
             f'reviews/questions_{candidate}.json',f'results/validation_{candidate}.json',
             f'reviews/validation_{candidate}.json']
    source = json.loads((r.PARENT/'results/verification.json').read_text(encoding='utf-8'))
    selection = {
        'id':candidate,'family':family,'spec':record['spec'],'details':record['details'],
        'model':r.e.CFG['model'],'revision':r.e.CFG['revision'],
        'generation':r.e.CFG,'edit_file':delta.relative_to(r.ROOT).as_posix(),
        'delta_sha256':sha256(delta),'heldout_sha256':sha256(r.ROOT/'data/heldout.json',text=True),
        'expected_source_checkpoint_sha256':source['source_checkpoint_sha256'],
        'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'target_style_rate':.5,
        'reviewed_development_metrics':reviewed,
        'selection_reason':'Corrected branch positions; layer 19 alpha 1 reached 56/100 natural endings and 27/50 validation, so no further edit was added.',
        'evaluation_role':'Existing questions and validation are development sets; held-out responses have not been generated.',
        'method':'Mean activation contrast, Gram-Schmidt projection and one outer-product down_proj update; no fitted loss or optimizer.',
        'file_hashes':{path:sha256(r.ROOT/path,text=True) for path in paths},
        'shared_file_hashes':{path:sha256(r.PARENT/path,text=True) for path in ['config.json','experiment.py','scoring.py']},
        'text_hash_policy':'UTF-8 with LF line endings',
    }
    r.e.dump(destination,selection)
    print(json.dumps({'id':candidate,'delta_sha256':selection['delta_sha256'],
                      'reviewed_development_metrics':reviewed},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',default='branch_04')
    parser.add_argument('--family',default='branch')
    args=parser.parse_args(); main(args.candidate,args.family)
