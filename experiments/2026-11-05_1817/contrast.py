"""Mechanistic hypothesis: instruction-conditioned direction before suffix choice.

Diagnostic reads compare identical answer prefixes. No regression/fitting,
output-layer edits, or diagnostic instruction is used in generated evaluation.
"""
import argparse
import torch
from safetensors.torch import save_file
import experiment as e

STYLE='ずんだもん風に、文末を「のだ」「なのだ」にして答えてください。'

@torch.inference_mode()
def collect_post(model,tok):
    stats=torch.load(e.ROOT/'runs/prefix_stats.pt',weights_only=True)
    records={i:[] for i in stats}
    for index,row in enumerate(e.read('train_pairs')):
        prefix=e.inputs(tok,[row['prompt']])['input_ids']
        answer=torch.tensor([tok.encode(row['style'],add_special_tokens=False)],device='cuda')
        x=torch.cat([prefix,answer],1); handles=[]; current={}
        for i,layer in enumerate(model.model.layers):
            def pre(m,args,i=i): current[i]=args[0][0,-3:].float().cpu()
            handles.append(layer.mlp.down_proj.register_forward_pre_hook(pre))
        try: model.model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False)
        finally:
            for h in handles: h.remove()
        for i in records: records[i].append(current[i])
        if (index+1)%20==0: print('post-suffix collect',index+1,flush=True)
    for i in records: stats[i]['post']=torch.stack(records[i])
    torch.save(stats,e.ROOT/'runs/prefix_stats_post.pt')

@torch.inference_mode()
def branch_odds(model,tok):
    seq=[]; labels=[]
    for row in e.read('train_pairs')[::5]:
        a=tok.encode(row['normal'],add_special_tokens=False); b=tok.encode(row['style'],add_special_tokens=False)
        j=0
        while j<min(len(a),len(b)) and a[j]==b[j]: j+=1
        if j==min(len(a),len(b)): continue
        prefix=e.inputs(tok,[row['prompt']])['input_ids'][0].tolist()
        seq.append(prefix+a[:j]); labels.append((a[j],b[j]))
    values=[]
    for start in range(0,len(seq),8):
        batch=tok.pad({'input_ids':seq[start:start+8]},padding=True,return_tensors='pt').to('cuda')
        logits=model(**batch,use_cache=False,logits_to_keep=1).logits[:,0].float()
        values.extend(float(v[b]-v[a]) for v,(a,b) in zip(logits,labels[start:start+8]))
    return sum(values)/len(values)

@torch.inference_mode()
def collect(model,tok):
    records={i:[] for i in range(len(model.model.layers))}
    for ix,row in enumerate(e.read('train_pairs')):
        normal=tok.encode(row['normal'],add_special_tokens=False)
        style=tok.encode(row['style'],add_special_tokens=False)
        common=[]
        for a,b in zip(normal,style):
            if a!=b: break
            common.append(a)
        paired=[]
        for instructed in [False,True]:
            messages=[{'role':'system','content':e.CFG['system']+(STYLE if instructed else '')},{'role':'user','content':row['prompt']}]
            prompt=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=True,return_tensors='pt')['input_ids'].to('cuda')
            x=torch.cat([prompt,torch.tensor([common],dtype=torch.long,device='cuda')],1)
            positions=[prompt.shape[1]-1,prompt.shape[1]+max(0,len(common)//2)-1,x.shape[1]-1]
            current={i:{} for i in records}; handles=[]
            for i,layer in enumerate(model.model.layers):
                def pre(m,args,i=i): current[i]['g']=args[0][0,positions].float().cpu()
                def post(m,args,out,i=i): current[i]['h']=(out[0] if isinstance(out,tuple) else out)[0,positions].float().cpu()
                handles.extend([layer.mlp.down_proj.register_forward_pre_hook(pre),layer.register_forward_hook(post)])
            try: model.model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False)
            finally:
                for handle in handles: handle.remove()
            paired.append(current)
        for i in records:
            records[i].append({'g':paired[0][i]['g'],'d':paired[1][i]['h']-paired[0][i]['h']})
        if (ix+1)%20==0: print('prefix collect',ix+1,flush=True)
    stats={}; summary={}
    for i,rr in records.items():
        g=torch.stack([r['g'] for r in rr]); d=torch.stack([r['d'] for r in rr])
        stats[i]={'g':g,'d':d}
        summary[i]={}
        for p,name in enumerate(['start','middle','branch']):
            delta=d[:,p].mean(0)
            summary[i][name]={'norm':float(delta.norm()),'coherence':float(delta.norm()/d[:,p].norm(dim=1).square().mean().sqrt())}
    torch.save(stats,e.ROOT/'runs/prefix_stats.pt')
    e.dump(e.ROOT/'results/prefix_summary.json',summary)
    return stats

def delta_for(stats,base,spec):
    edits={}
    for i in spec['layers']:
        p=spec.get('position',2); g=stats[i]['g'][:,p].mean(0); d=stats[i]['d'][:,p].mean(0); u=d/d.norm()
        w=base[i].float(); contribution=g*(u@w)
        # Suppress neurons that actively write against the desired direction.
        ranking=-contribution if spec['method']=='suppress' else contribution.abs()
        ids=ranking.topk(spec['columns']).indices; mask=torch.zeros_like(g); mask[ids]=1
        if spec['method']=='suppress': delta=-torch.outer(u,mask*(u@w))
        elif spec['method'] in ['gated_write','terminal_gated']:
            # Project away start/middle activity, without fitting a target loss.
            basis=[]
            nuisance=[stats[i]['g'][:,pos].mean(0) for pos in [0,1]]
            if spec['method']=='terminal_gated': nuisance.extend(stats[i]['post'][:,pos].mean(0) for pos in range(3))
            for vector in nuisance:
                v=vector.clone()
                for q in basis: v-=q*torch.dot(q,v)
                if v.norm()>1e-7: basis.append(v/v.norm())
            key=g.clone()
            for q in basis: key-=q*torch.dot(q,key)
            if g.norm()<1e-8 or key.norm()<g.norm()*.01: raise ValueError('Degenerate branch-selective key')
            delta=torch.outer(d,key/torch.dot(g,key))
        elif spec['method']=='write':
            key=g*mask; delta=torch.outer(d,key/key.square().sum().clamp_min(1e-8))
        else: raise ValueError(spec['method'])
        edits[f'model.layers.{i}.mlp.down_proj.weight']=delta*spec['alpha']/len(spec['layers'])
    return edits

@torch.inference_mode()
def search(model,tok,round_id,extension=False,terminal=False):
    stats=torch.load(e.ROOT/('runs/prefix_stats_post.pt' if terminal else 'runs/prefix_stats.pt'),weights_only=True)
    base={i:model.model.layers[i].mlp.down_proj.weight.detach().cpu().clone() for i in stats}
    originals={f'model.layers.{i}.mlp.down_proj.weight':v for i,v in base.items()}
    # Prefer late-layer coherent directions at the decision point.
    rank=lambda i: float(stats[i]['d'][:,2].mean(0).norm()/stats[i]['d'][:,2].norm(dim=1).square().mean().sqrt())
    ranked=sorted(range(12,24),key=lambda i:-rank(i))[:3]
    baseline_odds=branch_odds(model,tok)
    group=f'round{round_id}'+('_terminal' if terminal else '_extension' if extension else '')
    if round_id==2:
        specs=[{'method':method,'layers':[layer],'columns':256,'alpha':alpha,'position':2}
            for method in ['write','suppress'] for layer in ranked for alpha in [2.,6.]]
    else:
        # Odds improved in round 2, but continuous writes disrupted free generation.
        # Test a key orthogonal to mean start/middle activity at three depths.
        specs=[{'method':'gated_write','layers':layers,'columns':3584,'alpha':alpha,'position':2}
            for layers in [[15],[19],[23],[15,19,23]] for alpha in [.25,.5,1.]]
        if extension:
            specs=[{'method':'gated_write','layers':layers,'columns':3584,'alpha':alpha,'position':2}
                for layers in [[23],[15,19,23]] for alpha in [2.,4.,8.,16.]]
        if terminal:
            specs=[{'method':'terminal_gated','layers':layers,'columns':3584,'alpha':alpha,'position':2}
                for layers in [[23],[19,23]] for alpha in [1.,2.,4.,8.]]
        diagnostic={}
        for i in [15,19,23]:
            g=stats[i]['g'][:,2].mean(0); basis=[]
            nuisance=[stats[i]['g'][:,pos].mean(0) for pos in [0,1]]
            if terminal: nuisance.extend(stats[i]['post'][:,pos].mean(0) for pos in range(3))
            for vector in nuisance:
                v=vector.clone()
                for q in basis: v-=q*torch.dot(q,v)
                if v.norm()>1e-7: basis.append(v/v.norm())
            key=g.clone()
            for q in basis: key-=q*torch.dot(q,key)
            gains=stats[i]['g']@key/torch.dot(g,key)
            diagnostic[i]={'retained_key_fraction':float(key.norm()/g.norm()),
                'gain_quantiles_start_middle_branch':torch.quantile(gains,torch.tensor([.1,.5,.9]),dim=0).tolist(),
                'denominator':float(torch.dot(g,key)),'key_squared_norm':float(key.square().sum())}
            if terminal:
                post_gain=stats[i]['post']@key/torch.dot(g,key)
                diagnostic[i]['post_suffix_gain_quantiles']=torch.quantile(post_gain,torch.tensor([.1,.5,.9]),dim=0).tolist()
        e.dump(e.ROOT/('results/terminal_key_diagnostics.json' if terminal else 'results/gated_key_diagnostics.json'),diagnostic)
    val=e.read('validation'); screen=val[::5]; scores=[]
    for index,spec in enumerate(specs):
        name=f'{group}_{index:02d}'; edits=delta_for(stats,base,spec)
        e.apply(model,originals,edits)
        odds=branch_odds(model,tok)
        result=e.generate(model,tok,screen,e.ROOT/f'runs/{name}.json')
        from scoring import summarize
        metric=summarize(result['responses'])
        scores.append({'id':name,'spec':spec,'branch_log_odds':odds,'branch_log_odds_change':odds-baseline_odds,**metric})
        print(name,spec,metric,flush=True)
        e.dump(e.ROOT/f'results/{group}_progress.json',scores)
    top=sorted(scores,key=lambda s:(-s['heuristic_style_rate'],s['malformed_ending'],s['repetition']))[:3]
    for spec in top:
        edits=delta_for(stats,base,spec['spec']); e.apply(model,originals,edits)
        result=e.generate(model,tok,val,e.ROOT/f'results/validation_{spec["id"]}.json')
        spec['full']=summarize(result['responses'])
        save_file(edits,str(e.ROOT/f'models/{spec["id"]}_delta.safetensors'))
    e.apply(model,originals,{})
    e.dump(e.ROOT/f'results/{group}_search.json',{'ranked_layers':ranked,'base_branch_log_odds':baseline_odds,'screen':scores,'finalists':top})

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['collect','collect_post','search']); p.add_argument('--round',type=int,default=2)
    p.add_argument('--extension',action='store_true',help='Round 3: test stronger gated writes after the small-strength comparison')
    p.add_argument('--terminal',action='store_true',help='Round 3: also remove mean activity after the desired suffix')
    a=p.parse_args(); model,tok=e.load_model()
    if a.stage=='collect': collect(model,tok)
    elif a.stage=='collect_post': collect_post(model,tok)
    else: search(model,tok,a.round,a.extension,a.terminal)
