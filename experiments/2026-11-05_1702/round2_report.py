"""Summaries and figures derived from complete saved outputs, not new inference."""
import csv
import hashlib
import json
import os
import random
import sys
import numpy as np
import round2 as r
from scoring import score

os.environ.setdefault('MPLCONFIGDIR',str(r.ROOT/'cache/matplotlib'))
CONDITIONS=['base','prompt','edited','random']

def bootstrap(allrows):
    ids=sorted({x['id'] for x in allrows['base']}); counts={}
    for c,rr in allrows.items():
        counts[c]=np.array([[sum(score(x['text'])[key] for x in rr if x['id']==qid) for key in ['target_endings','sentences']] for qid in ids])
    rng=np.random.default_rng(492699); ix=rng.integers(0,len(ids),size=(10000,len(ids)))
    rates={c:(v[ix].sum(1)[:,0]/np.maximum(v[ix].sum(1)[:,1],1)) for c,v in counts.items()}
    return {f'{c}-base':dict(difference_pp=100*(r.metrics(allrows[c])['ending_rate']-r.metrics(allrows['base'])['ending_rate']),
        ci95_pp=(100*np.quantile(rates[c]-rates['base'],[.025,.975])).tolist()) for c in CONDITIONS[1:]}

def packets(partial=False):
    # First three fixed questions of each category, all three seeds, four conditions.
    picked={x['id'] for j,x in enumerate(r.data('test')) if j%12<3}
    mapping={}; rng=random.Random(81072)
    for index,seed in enumerate(r.e.CFG['final_seeds']):
        rr=[]
        for c in CONDITIONS:
            for row in r.rows(r.RUN/f'test_{c}.jsonl'):
                if row['id'] not in picked or row['seed']!=seed: continue
                blind=hashlib.sha256(f'round2:{c}:{row["id"]}:{seed}'.encode()).hexdigest()[:16]
                mapping[blind]=dict(condition=c,id=row['id'],seed=seed)
                rr.append(dict(id=blind,prompt=row['prompt'],text=row['text']))
        if partial and len(rr)!=48: continue
        assert len(rr)==48; rng.shuffle(rr); r.e.dump(r.RUN/f'blind_packet{index}.json',rr)
    r.e.dump(r.RUN/'blind_mapping.json',mapping)

def quality():
    mapping=json.loads((r.RUN/'blind_mapping.json').read_text(encoding='utf-8')); merged=[]
    for index in range(3):
        packet=json.loads((r.RUN/f'blind_packet{index}.json').read_text(encoding='utf-8'))
        ratings=json.loads((r.RUN/f'blind_reviews{index}.json').read_text(encoding='utf-8'))
        assert len(ratings)==48
        assert {x['review_id'] for x in ratings}=={x['id'] for x in packet}
        assert all(x['relevance'] in [0,1,2] and x['naturalness'] in [0,1,2] and x['note'] for x in ratings)
        merged.extend(dict(**x,**mapping[x['review_id']],reviewer=index) for x in ratings)
    summary={c:dict(n=len(rr),relevance=sum(x['relevance'] for x in rr)/len(rr),naturalness=sum(x['naturalness'] for x in rr)/len(rr))
        for c in CONDITIONS if (rr:=[x for x in merged if x['condition']==c])}
    assert all(s['n']==36 for s in summary.values())
    r.e.dump(r.RES/'quality_reviews.json',merged);r.e.dump(r.RES/'quality_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

def main():
    r.verify_inputs()
    for split in ['test','side']:
        expected={(x['id'],seed) for x in r.data(split) for seed in r.e.CFG['final_seeds']}
        for c in CONDITIONS:
            rr=r.rows(r.RUN/f'{split}_{c}.jsonl')
            assert len(rr)==len(expected) and {(x['id'],x['seed']) for x in rr}==expected
            assert all(x['condition']==c for x in rr)
    allrows={c:r.rows(r.RUN/f'test_{c}.jsonl') for c in CONDITIONS}
    assert all(len(rr)==144 for rr in allrows.values())
    summary={c:dict(test=r.metrics(rr),side=r.metrics(r.rows(r.RUN/f'side_{c}.jsonl'))) for c,rr in allrows.items()}
    r.e.dump(r.RES/'summary.json',summary)
    r.e.dump(r.RES/'bootstrap.json',bootstrap(allrows))
    baseline={(x['id'],x['seed']):x['text'] for x in allrows['base']}
    r.e.dump(r.RES/'text_agreement.json',{c:dict(n=len(rr),identical=sum(x['text']==baseline[(x['id'],x['seed'])] for x in rr))
        for c,rr in allrows.items() if c!='base'})
    r.e.dump(r.RES/'by_group.json',{c:{'category':{g:r.metrics([x for x in rr if x['category']==g]) for g in ['daily','knowledge','explanation','advice']},
        'seed':{seed:r.metrics([x for x in rr if x['seed']==seed]) for seed in r.e.CFG['final_seeds']}} for c,rr in allrows.items()})
    validation=json.loads((r.RES/'validation.json').read_text(encoding='utf-8'))
    final_table={c:dict(metrics=v['test']) for c,v in summary.items()}
    final_table['edited']['edit']=json.loads((r.RES/'verification.json').read_text(encoding='utf-8'))['edit']
    final_table['random']['edit']=json.loads((r.RES/'random_edit.json').read_text(encoding='utf-8'))
    for name,table in [('validation',validation),('final',final_table)]:
        records=[dict(condition=c,**v['metrics'],changed=v.get('edit',{}).get('changed',0),delta_norm=v.get('edit',{}).get('norm',0)) for c,v in table.items()]
        with (r.RES/f'{name}.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    records=[dict(condition=c,id=x['id'],seed=x['seed'],**score(x['text']),tokens=x['tokens'],hit_limit=x['hit_limit']) for c,rr in allrows.items() for x in rr]
    with (r.RES/'per_response_metrics.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    # Examples fixed by question position, not success/failure after seeing results.
    exampleids={x['id'] for j,x in enumerate(r.data('test')) if j%12==0}
    r.e.dump(r.RES/'examples.json',[{k:x[k] for k in ['condition','id','seed','prompt','text']} for rr in allrows.values() for x in rr if x['id'] in exampleids and x['seed']==101])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    image=r.ROOT/'images/weight-space-zundamon'; image.mkdir(exist_ok=True)
    fig,ax=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    mat=np.array([[100*validation[f'write_{g}_a{a:g}']['metrics']['ending_rate'] for a in [.5,2,6]] for g in ['15','19','both']])
    im=ax[0].imshow(mat,cmap='YlGn',vmin=0,vmax=max(1,mat.max()))
    ax[0].set_xticks(range(3),['0.5','2','6']);ax[0].set_yticks(range(3),['Layer 15','Layer 19','Layers 15 + 19'])
    ax[0].set_xlabel('Intervention strength alpha');ax[0].set_title('Validation: target sentence endings (%)')
    for i in range(3):
        for j in range(3): ax[0].text(j,i,f'{mat[i,j]:.1f}',ha='center',va='center')
    values=[100*summary[c]['test']['ending_rate'] for c in CONDITIONS]
    bars=ax[1].bar(CONDITIONS,values,color=['#888888','#dca342','#3a985d','#6c86b7'])
    ax[1].bar_label(bars,fmt='%.1f%%',padding=3);ax[1].set_ylim(0,max(5,max(values)*1.3));ax[1].set_ylabel('Target sentence endings (%)')
    ax[1].set_title('Held-out: 48 questions x 3 seeds')
    fig.savefig(image/'round2-comparison.png',dpi=180);plt.close(fig)
    files=list(r.RUN.glob('*.jsonl'))+[r.RUN/'stats.pt',r.ROOT/'models/round2_delta.safetensors']
    r.e.dump(r.RES/'provenance.json',dict(model=r.e.CFG,files={str(p.relative_to(r.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        generation_count=sum(len(r.rows(p)) for p in r.RUN.glob('*.jsonl')),
        generation_seconds=sum(x['seconds'] for p in r.RUN.glob('*.jsonl') for x in r.rows(p))))
    packets()
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    if '--quality' in sys.argv: quality()
    elif '--packets' in sys.argv: packets(partial=True)
    else: main()
