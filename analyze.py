"""Aggregate final results and export randomized, condition-blind review packets."""
import os
import json
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'cache/matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scoring import score, validate_side
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
OUT.mkdir(exist_ok=True)
def read(split,condition):
    return [json.loads(s) for s in (ROOT/'runs'/f'{split}_{condition}.jsonl').read_text(encoding='utf-8').splitlines()]
def aggregate(rows,side=False):
    ms=[score(r['text']) for r in rows]
    result={'responses':len(rows),'sentences':sum(m['sentences'] for m in ms),
        'target_endings':sum(m['target_endings'] for m in ms),
        'mention_rate':np.mean([m['target_mention'] for m in ms]).item(),
        'degenerate':sum(m['empty'] or m['style_only'] for m in ms),
        'repetition':sum(m['repetition'] for m in ms),
        'mean_tokens':np.mean([r['tokens'] for r in rows]).item(),
        'cap_hits':sum(r['hit_limit'] for r in rows),
        'seconds':sum(r['seconds'] for r in rows)}
    result['ending_rate']=result['target_endings']/max(1,result['sentences'])
    if side:
        result['instruction_pass_rate']=np.mean([validate_side(r['text'],r['validator']) for r in rows]).item()
    return result
def main():
    result={}
    allrows={}
    for split in ['test','side']:
        result[split]={}
        for cond in ['base','prompt','edited']:
            rows=read(split,cond)
            expected=(48 if split=='test' else 16)*3
            assert len(rows)==expected and len({(r['id'],r['seed']) for r in rows})==expected
            result[split][cond]=aggregate(rows,split=='side')
            if split=='test': allrows[cond]=rows
    ids=sorted({r['id'] for r in allrows['base']})
    result['by_category']={c:{cat:aggregate([r for r in rows if r['category']==cat])
        for cat in sorted({r['category'] for r in rows})} for c,rows in allrows.items()}
    result['by_seed']={c:{str(seed):aggregate([r for r in rows if r['seed']==seed])
        for seed in sorted({r['seed'] for r in rows})} for c,rows in allrows.items()}
    base_text={(r['id'],r['seed']):r['text'] for r in allrows['base']}
    result['descriptive_edited_identical_to_base_rate']=sum(
        r['text']==base_text[(r['id'],r['seed'])] for r in allrows['edited'])/len(allrows['edited'])
    counts={c:np.array([[sum(score(r['text'])[key] for r in rs if r['id']==qid)
        for key in ['target_endings','sentences']] for qid in ids]) for c,rs in allrows.items()}
    rng=np.random.default_rng(20261105)
    diffs=[]
    for _ in range(10000):
        ix=rng.integers(0,len(ids),len(ids))
        a=counts['edited'][ix].sum(0); b=counts['base'][ix].sum(0)
        diffs.append(a[0]/max(1,a[1])-b[0]/max(1,b[1]))
    result['edited_minus_base_95ci']=np.quantile(diffs,[.025,.975]).tolist()
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    labels=['Base','Style prompt','Weight edit']
    fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    axes[0].bar(labels,[100*result['test'][c]['ending_rate'] for c in allrows],color=['#607d8b','#4caf50','#e69138'])
    axes[0].set(ylabel='Target sentence endings (%)',ylim=(0,100),title='Style without runtime steering')
    axes[1].bar(labels,[100*result['side'][c]['instruction_pass_rate'] for c in allrows],color=['#607d8b','#4caf50','#e69138'])
    axes[1].set(ylabel='Explicit instruction checks passed (%)',ylim=(0,100),title='Side effects (16 prompts × 3 seeds)')
    for ax in axes:
        for bar in ax.patches:
            ax.text(bar.get_x()+bar.get_width()/2, min(97,bar.get_height()+2),
                f'{bar.get_height():.1f}%',ha='center',va='bottom',fontsize=11)
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)
    from make_blind_packets import export
    assert export()==[0,1,2]
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
