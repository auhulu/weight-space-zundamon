"""Build numeric result tables; article prose is reviewed separately."""
import csv
import hashlib
import json
import shutil
from pathlib import Path
from scoring import score
ROOT=Path(__file__).resolve().parent
def main():
    out=ROOT/'results';out.mkdir(exist_ok=True)
    for name in ['selection.json','verification.json','environment.json','candidate_specs.json','stats_summary.json']:
        shutil.copyfile(ROOT/'runs'/name,out/name)
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    with (out/'metrics.csv').open('w',encoding='utf-8',newline='') as f:
        fields=['split','condition',*summary['test']['base'].keys(),'instruction_pass_rate']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for split in ['test','side']:
            for condition,metrics in summary[split].items():writer.writerow({'split':split,'condition':condition,**metrics})
    records=[]
    for condition in ['base','prompt','edited']:
        rows=[json.loads(s) for s in (ROOT/'runs'/f'test_{condition}.jsonl').read_text(encoding='utf-8').splitlines()]
        records.extend({'id':r['id'],'seed':r['seed'],'condition':condition,'category':r['category'],**score(r['text'])} for r in rows)
    with (out/'per_response_metrics.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    mapping=json.loads((ROOT/'runs'/'blind_mapping.json').read_text())
    reviews=[]
    for i in range(3):reviews.extend(json.loads((ROOT/'runs'/'blind'/f'reviews_{i}.json').read_text(encoding='utf-8')))
    assert len(reviews)==len(mapping) and {r['review_id'] for r in reviews}==set(mapping)
    for r in reviews:
        assert r['relevance'] in [0,1,2] and r['naturalness'] in [0,1,2]
        r.update(mapping[r['review_id']])
    quality={}
    for c in ['base','prompt','edited']:
        rs=[r for r in reviews if r['condition']==c]
        quality[c]={k:sum(r[k] for r in rs)/len(rs) for k in ['relevance','naturalness']}
        quality[c]['both_2_rate']=sum(r['relevance']==2 and r['naturalness']==2 for r in rs)/len(rs)
        quality[c]['n']=len(rs)
    (out/'quality_summary.json').write_text(json.dumps(quality,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'quality_reviews.json').write_text(json.dumps(reviews,ensure_ascii=False,indent=2),encoding='utf-8')
    raw_paths=sorted((ROOT/'runs').glob('*.jsonl'))
    generation_rows=[json.loads(s) for p in raw_paths for s in p.read_text(encoding='utf-8').splitlines()]
    paths=[*raw_paths,ROOT/'experiment.py',ROOT/'edit_design.py',ROOT/'scoring.py',ROOT/'config.json',ROOT/'requirements.lock']
    manifest={'generation_count':len(generation_rows),'generation_seconds':sum(r['seconds'] for r in generation_rows),
        'sha256':{str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(quality,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
