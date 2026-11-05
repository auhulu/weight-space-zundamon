"""Continuation: mechanism-guided editing with no gradients or fitted losses."""
import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent
sys.path.insert(0,str(PARENT))
import experiment as e
import torch
from safetensors.torch import save_file,load_file
from scoring import summarize

EXAMPLES=[
 ('一分は何秒？','一分は60秒だ。','一分は60秒なのだ。'),
 ('晴れた昼の空はどんな色？','晴れた昼の空は青い。','晴れた昼の空は青いのだ。'),
 ('待ち合わせに遅れない工夫は？','少し早めに家を出る。','少し早めに家を出るのだ。'),
 ('積み木を片づけるには？','同じ形の積み木を箱にまとめる。','同じ形の積み木を箱にまとめるのだ。'),
 ('疲れたときはどうする？','無理をせず少し休む。','無理をせず少し休むのだ。'),
 ('朝に飲みたいものは？','温かいお茶が好きだ。','温かいお茶が好きなのだ。'),
 ('雪の日に外へ出るときは？','暖かい上着を着る。','暖かい上着を着るのだ。'),
 ('友達に助けてもらったら？','ありがとうと伝えると嬉しい。','ありがとうと伝えると嬉しいのだ。'),
]
STYLE='ずんだもん風に、すべての回答の文末を「のだ」または「なのだ」にしてください。'
def read(name): return json.loads((ROOT/'data'/f'{name}.json').read_text(encoding='utf-8'))
def messages(question,mode='base'):
    system=e.CFG['system']
    if mode=='instruction': system+=STYLE
    if mode=='english': system+=' Always end your Japanese answer with のだ or なのだ. Give one complete short sentence only.'
    if mode=='few_instruction': system+=STYLE
    out=[{'role':'system','content':system}]
    if mode.startswith('few'):
        n=4 if mode.endswith('4') else 8
        style=not mode.startswith('few_neutral')
        for q,a,b in EXAMPLES[:n]: out.extend([{'role':'user','content':q},{'role':'assistant','content':b if style else a}])
    out.append({'role':'user','content':question})
    return out

@torch.inference_mode()
def generate(model,tok,questions,path,mode='base'):
    started=perf_counter(); rr=[]
    eos=model.generation_config.eos_token_id; eos={eos} if isinstance(eos,int) else set(eos)
    for start in range(0,len(questions),8):
        batch=questions[start:start+8]
        x=tok.apply_chat_template([messages(r['prompt'],mode) for r in batch],tokenize=True,
            add_generation_prompt=True,enable_thinking=False,padding=True,return_dict=True,return_tensors='pt').to('cuda')
        outputs=model.generate(**x,max_new_tokens=128,do_sample=False,pad_token_id=tok.pad_token_id,use_cache=True)
        for r,ids in zip(batch,outputs[:,x['input_ids'].shape[1]:].tolist()):
            end=next((i+1 for i,t in enumerate(ids) if t in eos),len(ids)); ids=ids[:end]
            rr.append({**r,'answer':tok.decode(ids,skip_special_tokens=True).strip(),'tokens':len(ids),'hit_limit':len(ids)==128 and eos.isdisjoint(ids)})
    result={**e.CFG,'mode':mode,'diagnostic_only':mode!='base','seconds':round(perf_counter()-started,2),'responses':rr}
    e.dump(path,result); return result

def diagnostic(model,tok):
    q=e.read('validation')[::5]
    for mode in ['instruction','english','few_style4','few_style8','few_instruction','few_neutral8']:
        result=generate(model,tok,q,ROOT/f'results/diagnostic_{mode}.json',mode)
        print(mode,summarize(result['responses']),flush=True)

@torch.inference_mode()
def collect(model,tok):
    records={i:[] for i in range(len(model.model.layers))}
    modes=['base','few_neutral8','few_instruction']
    for ix,row in enumerate(read('pairs')):
        a=tok.encode(row['normal'],add_special_tokens=False); b=tok.encode(row['style'],add_special_tokens=False)
        common=0
        while common<min(len(a),len(b)) and a[common]==b[common]: common+=1
        paired=[]
        for mode in modes:
            prompt=tok.apply_chat_template(messages(row['prompt'],mode),tokenize=True,add_generation_prompt=True,
                enable_thinking=False,return_dict=True,return_tensors='pt')['input_ids'].to('cuda')
            p=prompt.shape[1]; x=torch.cat([prompt,torch.tensor([b],device='cuda')],1)
            pos=[p-3,p-2,p-1,p,p+max(1,common//2)-1,p+common-1,x.shape[1]-3,x.shape[1]-2,x.shape[1]-1]
            current={i:{} for i in records}; handles=[]
            for i,layer in enumerate(model.model.layers):
                def pre(m,args,i=i): current[i]['g']=args[0][0,pos].float().cpu()
                def post(m,args,out,i=i): current[i]['h']=(out[0] if isinstance(out,tuple) else out)[0,pos].float().cpu()
                handles.extend([layer.mlp.down_proj.register_forward_pre_hook(pre),layer.register_forward_hook(post)])
            try: model.model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False)
            finally:
                for handle in handles: handle.remove()
            paired.append(current)
        for i in records:
            records[i].append({'g':paired[0][i]['g'],'d_style':paired[2][i]['h']-paired[1][i]['h'],
                'd_total':paired[2][i]['h']-paired[0][i]['h']})
        if (ix+1)%10==0: print('collect',ix+1,flush=True)
    stats={i:{k:torch.stack([r[k] for r in rr]) for k in ['g','d_style','d_total']} for i,rr in records.items()}
    torch.save(stats,ROOT/'runs/fewshot_stats.pt')
    summary={}
    for i,s in stats.items():
        summary[i]={}
        for mode in ['d_style','d_total']:
            d=s[mode][:,2]; summary[i][mode]={'start_norm':float(d.mean(0).norm()),
                'start_coherence':float(d.mean(0).norm()/d.norm(dim=1).square().mean().sqrt())}
    e.dump(ROOT/'results/activation_summary.json',summary)

def orthogonal_key(g,negative):
    if not torch.isfinite(g).all() or g.norm()<1e-8: raise ValueError('Degenerate key')
    basis=[]
    for vector in negative:
        if not torch.isfinite(vector).all(): raise ValueError('Nonfinite nuisance activity')
        v=vector.clone()
        for q in basis: v-=q*torch.dot(q,v)
        if v.norm()>1e-7: basis.append(v/v.norm())
    key=g.clone()
    for q in basis: key-=q*torch.dot(q,key)
    if not torch.isfinite(key).all() or key.norm()<g.norm()*.01: raise ValueError('Degenerate key')
    return key

def make_edit(stats,spec):
    edits={}; details={}
    for i in spec['layers']:
        s=stats[i]; pos=spec.get('position',2); positions=spec.get('positions',[pos])
        g=s['g'][:,positions].mean((0,1))
        neg=spec.get('negative',[0,1,3,4,5,6,7,8])
        key=orthogonal_key(g,[s['g'][:,j].mean(0) for j in neg])
        d=s[spec['direction']][:,positions].mean((0,1))
        delta=spec['alpha']/len(spec['layers'])*torch.outer(d,key/torch.dot(g,key))
        assert torch.isfinite(delta).all()
        edits[f'model.layers.{i}.mlp.down_proj.weight']=delta
        gain=s['g']@key/torch.dot(g,key)
        details[i]={'key_fraction':float(key.norm()/g.norm()),'delta_norm':float(delta.norm()),
            'gain_quantiles':torch.quantile(gain,torch.tensor([.1,.5,.9]),dim=0).tolist()}
    return edits,details

@torch.inference_mode()
def search(model,tok,family='start'):
    stats=torch.load(ROOT/'runs/fewshot_stats.pt',weights_only=True)
    originals={f'model.layers.{i}.mlp.down_proj.weight':model.model.layers[i].mlp.down_proj.weight.detach().cpu().clone() for i in stats}
    specs=[{'layers':layers,'direction':direction,'alpha':alpha} for direction in ['d_style','d_total']
        for layers in [[7],[11],[7,11]] for alpha in [1.,3.]]
    if family=='sustain':
        specs=[{'layers':layers,'direction':'d_style','alpha':alpha,'positions':[3,4,5],'negative':negative}
            for negative in [[],[0,1,2,6,7,8]] for layers in [[11],[15]] for alpha in [2.,6.,12.]]
    if family=='branch':
        # The suffix is one token: legacy position 6 equals branch, not post-suffix.
        specs=[{'layers':layers,'direction':'d_style','alpha':alpha,'position':5,
                'negative':[0,1,2,3,4,7,8]}
            for layers in [[15],[19],[23]] for alpha in [1.,3.,6.,12.]]
    q=e.read('validation')[::5]; scores=[]
    for index,spec in enumerate(specs):
        name=f'{family}_{index:02d}'; delta,details=make_edit(stats,spec)
        e.apply(model,originals,delta)
        result=generate(model,tok,q,ROOT/f'results/{name}.json')
        metric=summarize(result['responses']); record={'id':name,'spec':spec,'details':details,**metric}
        scores.append(record); e.dump(ROOT/f'results/{family}_search.json',scores)
        save_file(delta,str(ROOT/f'models/{name}_delta.safetensors'))
        print(name,spec,metric,flush=True)
    e.apply(model,originals,{})

def main():
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['diagnostic','collect','search','evaluate'])
    p.add_argument('--family',default='start',choices=['start','sustain','branch'])
    p.add_argument('--candidates',nargs='+')
    p.add_argument('--split',default='validation',choices=['validation','questions'])
    a=p.parse_args()
    for directory in ['data','runs','results','models','reviews']: (ROOT/directory).mkdir(exist_ok=True)
    model,tok=e.load_model()
    if a.stage=='diagnostic': diagnostic(model,tok)
    elif a.stage=='collect': collect(model,tok)
    elif a.stage=='search': search(model,tok,a.family)
    elif a.stage=='evaluate':
        if not a.candidates: p.error('evaluate requires --candidates')
        originals={f'model.layers.{i}.mlp.down_proj.weight':layer.mlp.down_proj.weight.detach().cpu().clone()
            for i,layer in enumerate(model.model.layers)}
        for name in a.candidates:
            edits=load_file(str(ROOT/f'models/{name}_delta.safetensors'))
            e.apply(model,originals,edits)
            path=ROOT/f'results/{a.split}_{name}.json'
            if path.exists(): raise FileExistsError(path)
            result=generate(model,tok,e.read(a.split),path)
            result.update(candidate=name,evaluation_role='development')
            e.dump(path,result)
            print(a.split,name,summarize(result['responses']),flush=True)

if __name__=='__main__': main()
