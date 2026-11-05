"""Second frozen experiment: matched continuations, sparse writes and controls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch
from safetensors.torch import save_file
import experiment as e
from scoring import score, validate_side

ROOT=e.ROOT; DATA=ROOT/'data/round2'; RUN=ROOT/'runs/round2'; RES=ROOT/'results/round2'
RUN.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
LAYERS=[15,19]

def data(split): return json.loads((DATA/f'{split}.json').read_text(encoding='utf-8'))
def rows(path): return e.read_rows(path) if path.exists() else []
def verify_inputs():
    manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    for p,h in manifest.items():
        assert hashlib.sha256((ROOT/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h,p
    frozen=RES/'implementation.json'
    hashes={p:hashlib.sha256((ROOT/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in ['round2.py','experiment.py','scoring.py','config.json']}
    if frozen.exists(): assert json.loads(frozen.read_text())==hashes,'Implementation changed after freeze'
    else: e.dump(frozen,hashes)

def metrics(rr):
    ss=[score(r['text']) for r in rr]; n=len(rr)
    total=sum(s['sentences'] for s in ss); ends=sum(s['target_endings'] for s in ss)
    return dict(n=n,sentences=total,target_endings=ends,ending_rate=ends/max(total,1),
        mention_rate=sum(s['target_mention'] for s in ss)/max(n,1),
        degenerate=sum(s['empty'] or s['style_only'] for s in ss),
        repetition=sum(s['repetition'] for s in ss),hit_limit=sum(r['hit_limit'] for r in rr),
        mean_tokens=sum(r['tokens'] for r in rr)/max(n,1),
        side_pass=sum(validate_side(r['text'],r['validator']) for r in rr if 'validator' in r))

def evaluate(model,tok,split,condition,seeds):
    path=RUN/f'{split}_{condition}.jsonl'; done={(r['id'],r['seed']) for r in rows(path)}
    for seed in seeds:
        for item in data(split):
            if (item['id'],seed) in done: continue
            out=e.generate(model,tok,item['prompt'],seed,style=condition=='prompt')
            row={**item,**out,'condition':condition,'seed':seed}
            with path.open('a',encoding='utf-8') as f: f.write(json.dumps(row,ensure_ascii=False)+'\n')
            print(f'{split} {condition} {item["id"]} seed={seed} tokens={out["tokens"]} seconds={out["seconds"]:.1f}',flush=True)
    return rows(path)

@torch.inference_mode()
def collect(model,tok):
    path=RUN/'stats.pt'
    if path.exists(): return torch.load(path,weights_only=True)
    records={i:[] for i in LAYERS}
    for item in data('train'):
        paired=[]
        for style in [False,True]:
            prefix=e.inputs(tok,item['prompt'])['input_ids']
            ans=tok(item['style' if style else 'normal'],add_special_tokens=False,return_tensors='pt')['input_ids'].to('cuda')
            x=torch.cat([prefix,ans],dim=1); width=min(24,ans.shape[1]); current={i:{} for i in LAYERS}; handles=[]
            for i in LAYERS:
                def pre(m,args,i=i): current[i]['g']=args[0][0,-width:].float().mean(0).cpu()
                def post(m,args,out,i=i):
                    if isinstance(out,tuple): out=out[0]
                    current[i]['h']=out[0,-width:].float().mean(0).cpu()
                layer=model.model.layers[i]
                handles += [layer.mlp.down_proj.register_forward_pre_hook(pre),layer.register_forward_hook(post)]
            try: model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,logits_to_keep=1)
            finally:
                for handle in handles: handle.remove()
            paired.append(current)
        for i in LAYERS:
            records[i].append(dict(g=paired[0][i]['g'],dg=paired[1][i]['g']-paired[0][i]['g'],d=paired[1][i]['h']-paired[0][i]['h']))
        print('collect '+item['id'],flush=True)
    stats={}
    for i,rr in records.items():
        g=torch.stack([r['g'] for r in rr]).mean(0); dg=torch.stack([r['dg'] for r in rr]).mean(0)
        dd=torch.stack([r['d'] for r in rr]); d=dd.mean(0); u=d/d.norm()
        w=model.model.layers[i].mlp.down_proj.weight.float().cpu()
        writer=dg*(u@w); indices=writer.topk(64).indices
        mask=torch.zeros_like(g); mask[indices]=1; key=g*mask
        rng=torch.Generator().manual_seed(492698+i)
        random=torch.randn(d.shape,generator=rng); random-=u*torch.dot(u,random); random/=random.norm()
        stats[i]=dict(g=g,dg=dg,d=d,u=u,key=key,mask=mask,random=random,writer=writer,
            indices=indices,coherence=d.norm()/dd.norm(dim=1).square().mean().sqrt())
    torch.save(stats,path)
    e.dump(RES/'activation_summary.json',{i:dict(d_norm=float(s['d'].norm()),key_norm=float(s['key'].norm()),
        coherence=float(s['coherence']),columns=s['indices'].tolist(),positive_selected=int((s['writer'][s['indices']]>0).sum()),
        random_cosine=float(s['random']@s['u'])) for i,s in stats.items()})
    return stats

def specs():
    out=[dict(id=f'write_{name}_a{a:g}',layers=layers,alpha=a,method='write') for name,layers in [('15',[15]),('19',[19]),('both',LAYERS)] for a in [.5,2.,6.]]
    out.append(dict(id='project_both_a6',layers=LAYERS,alpha=6.,method='project'))
    out.extend(dict(id=f'random_both_a{a:g}',layers=LAYERS,alpha=a,method='write',random=True) for a in [.5,2.,6.])
    return out

def originals(model): return {i:model.model.layers[i].mlp.down_proj.weight.detach().cpu().clone() for i in LAYERS}
@torch.no_grad()
def restore(model,base):
    for i,w in base.items(): model.model.layers[i].mlp.down_proj.weight.copy_(w)

@torch.no_grad()
def apply(model,base,stats,spec):
    restore(model,base); details=[]
    for i in spec['layers']:
        s=stats[i]; w=base[i].float()
        if spec['method']=='write':
            left=s['d']; right=s['key']/s['key'].square().sum()
        else:
            left=s['u']; right=s['mask']*(s['u']@w)
        if spec.get('random'): left=s['random']*left.norm()
        delta=spec['alpha']*torch.outer(left,right)
        edited=(w+delta).half(); assert torch.isfinite(edited).all()
        model.model.layers[i].mlp.down_proj.weight.copy_(edited)
        actual=edited.float()-w
        details.append(dict(layer=i,changed=int((edited!=base[i]).sum()),planned_norm=float(delta.norm()),
            actual_norm=float(actual.norm()),relative_norm=float(actual.norm()/w.norm())))
    return dict(spec=spec,layers=details,changed=sum(d['changed'] for d in details),
        norm=sum(d['actual_norm']**2 for d in details)**.5)

def select():
    table=json.loads((RES/'validation.json').read_text(encoding='utf-8')); base=table['base']['metrics']
    candidates=[]
    for spec in specs():
        if spec.get('random'): continue
        r=table[spec['id']]; m=r['metrics']
        excess=sum(max(0,m[k]-base[k]) for k in ['degenerate','repetition','hit_limit'])
        candidates.append((excess,-m['ending_rate'],r['edit']['norm'],spec))
    eligible=[c for c in candidates if c[0]==0]
    best=min(eligible or candidates,key=lambda x:x[:3]); chosen=best[3]
    result=dict(spec=chosen,eligible=best[0]==0,validation=table[chosen['id']],note='Selection frozen before final outputs')
    p=RES/'selection.json'
    if p.exists(): assert json.loads(p.read_text(encoding='utf-8'))==result
    else: e.dump(p,result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result

def hashes(model): return {n:e.tensor_digest(p) for n,p in model.named_parameters()}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['search','final']); args=parser.parse_args()
    verify_inputs(); torch.set_num_threads(6); model,tok=e.load_model(); base=originals(model)
    stats=collect(model,tok)
    if args.stage=='search':
        table={}
        for c in ['base','prompt']:
            table[c]=dict(metrics=metrics(evaluate(model,tok,'validation',c,[42])))
            e.dump(RES/'validation.json',table)
        for spec in specs():
            detail=apply(model,base,stats,spec)
            rr=evaluate(model,tok,'validation',spec['id'],[42]); table[spec['id']]=dict(metrics=metrics(rr),edit=detail)
            e.dump(RES/'validation.json',table)
        restore(model,base); select()
    else:
        selection=select(); spec=selection['spec']; before=hashes(model)
        detail=apply(model,base,stats,spec); after=hashes(model)
        changed=[n for n in before if before[n]!=after[n]]
        expected=[f'model.layers.{i}.mlp.down_proj.weight' for i in spec['layers']]
        assert sorted(changed)==sorted(expected)
        assert all(torch.isfinite(p).all() for p in model.parameters())
        dest=ROOT/'models/round2-edited'; model.save_pretrained(dest,safe_serialization=True); tok.save_pretrained(dest)
        save_file({f'model.layers.{i}.mlp.down_proj.weight':model.model.layers[i].mlp.down_proj.weight.float().cpu()-base[i].float() for i in spec['layers']},str(ROOT/'models/round2_delta.safetensors'))
        probe=e.generate(model,tok,data('train')[0]['prompt'],101,greedy=True)['token_ids']
        restore(model,base); assert hashes(model)==before
        del model; torch.cuda.empty_cache()
        model,tok=e.load_model(str(dest)); assert hashes(model)==after
        assert e.generate(model,tok,data('train')[0]['prompt'],101,greedy=True)['token_ids']==probe
        e.dump(RES/'verification.json',dict(changed=changed,unchanged=len(before)-len(changed),
            all_finite=True,restored_exact=True,reload_all_hashes_exact=True,reload_greedy_exact=True,edit=detail,
            before={n:before[n] for n in changed},after={n:after[n] for n in changed}))
        table={}
        for condition in ['edited','base','prompt','random']:
            if condition in ['base','prompt']: restore(model,base)
            if condition=='random':
                random_detail=apply(model,base,stats,{**spec,'random':True})
                e.dump(RES/'random_edit.json',random_detail)
            table[condition]={}
            for split in ['test','side']:
                rr=evaluate(model,tok,split,condition,e.CFG['final_seeds']); table[condition][split]=metrics(rr)
            e.dump(RES/'summary.json',table)

if __name__=='__main__': main()
