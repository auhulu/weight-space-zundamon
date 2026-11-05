"""Select from validation only. Never reads final data."""
import json
from pathlib import Path
from scoring import score
R=Path(__file__).resolve().parent/'runs'
def summarize(condition):
    rows=[json.loads(x) for x in (R/f'validation_{condition}.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==24 and len({r['id'] for r in rows})==24
    metrics=[score(r['text']) for r in rows]
    return {'ending_rate':sum(m['target_endings'] for m in metrics)/max(1,sum(m['sentences'] for m in metrics)),
        'degenerate':sum(m['empty'] or m['style_only'] for m in metrics),
        'repetition':sum(m['repetition'] for m in metrics),
        'target_mention_rate':sum(m['target_mention'] for m in metrics)/len(metrics)}
def main():
    specs=json.loads((R/'candidate_specs.json').read_text())
    base=summarize('base')
    results=[]
    for spec in specs:
        metrics=summarize(spec['id'])
        edit=json.loads((R/f"edit_{spec['id']}.json").read_text())
        count=edit.get('changed_elements', edit.get('total_changed_elements'))
        if count is None: raise ValueError('editor metadata must include changed_elements')
        results.append({'spec':spec, 'metrics':metrics,'changed_elements':count,
            'eligible':metrics['degenerate']<=base['degenerate'] and metrics['repetition']<=base['repetition']})
    eligible=[r for r in results if r['eligible']]
    best=sorted(eligible or results,key=lambda r:(-r['metrics']['ending_rate'],r['changed_elements']))[0]
    result={**best,'selection_passed':bool(eligible),'base':base,'prompt':summarize('prompt'),'candidates':results}
    (R/'selection.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
