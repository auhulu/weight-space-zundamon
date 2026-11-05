"""Refine the 56% result through observed internal MLP interventions only."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent
PREVIOUS=PARENT/'continuation_50'
sys.path.insert(0,str(PREVIOUS))
import run as prior
import torch
from safetensors.torch import save_file,load_file
e=prior.e

def stats():
    return torch.load(PREVIOUS/'runs/fewshot_stats.pt',map_location='cpu',weights_only=True)

def specifications(family):
    common={'direction':'d_style','position':5,'negative':[0,1,2,3,4,7,8]}
    if family=='strength':
        return [{**common,'layers':[19],'alpha':alpha} for alpha in [.75,1.25,1.5,2.]]+[
            {**common,'layers':[layer],'alpha':1.} for layer in [18,20,21]]
    if family=='class':
        return [{'kind':'class','layers':[19],'alpha':alpha,'class_nuisance':separate}
                for separate in [False,True] for alpha in [1.,1.25]]
    if family=='post':
        return [{'kind':'post','base_alpha':1.25,'post_layer':19,'post_alpha':alpha}
                for alpha in [.5,1.,2.]]
    if family=='class_post':
        return [{'kind':'class_post','layers':[19],'alpha':alpha,'class_nuisance':False,
                 'post_layer':19,'post_alpha':2.} for alpha in [1.,1.25]]
    raise ValueError(family)

def make_edit(observed,spec):
    if spec.get('kind')=='class':
        from class_edit import make_edit as class_edit
        return class_edit(observed,spec)
    if spec.get('kind')=='class_post':
        from class_edit import make_edit as class_edit
        edits,base_details=class_edit(observed,spec)
        extra,post_details=prior.make_edit(observed,{'layers':[spec['post_layer']],
            'direction':'d_style','position':7,'negative':[0,1,2,3,4,5,8],'alpha':spec['post_alpha']})
        for name,delta in extra.items(): edits[name]=edits.get(name,torch.zeros_like(delta))+delta
        return edits,{'base':base_details,'post':post_details}
    if spec.get('kind')=='post':
        base={'layers':[19],'direction':'d_style','position':5,
              'negative':[0,1,2,3,4,7,8],'alpha':spec['base_alpha']}
        post={'layers':[spec['post_layer']],'direction':'d_style','position':7,
              'negative':[0,1,2,3,4,5,8],'alpha':spec['post_alpha']}
        edits,base_details=prior.make_edit(observed,base)
        extra,post_details=prior.make_edit(observed,post)
        for name,delta in extra.items(): edits[name]=edits.get(name,torch.zeros_like(delta))+delta
        return edits,{'base':base_details,'post':post_details}
    return prior.make_edit(observed,spec)

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

@torch.inference_mode()
def search(model,tok,family):
    observed=stats()
    originals={f'model.layers.{i}.mlp.down_proj.weight':layer.mlp.down_proj.weight.detach().cpu().clone()
        for i,layer in enumerate(model.model.layers)}
    questions=e.read('validation')[::2]
    records=[]
    for ix,spec in enumerate(specifications(family)):
        name=f'{family}_{ix:02d}'; path=ROOT/f'runs/screen_{name}.json'
        if path.exists(): raise FileExistsError(path)
        delta,details=make_edit(observed,spec)
        e.apply(model,originals,delta)
        delta_path=ROOT/f'models/{name}_delta.safetensors'
        save_file(delta,str(delta_path))
        generated=prior.generate(model,tok,questions,path)
        metric=prior.summarize(generated['responses'])
        record={'id':name,'spec':spec,'details':details,'delta_sha256':sha(delta_path),**metric}
        records.append(record); e.dump(ROOT/f'results/{family}_search.json',records)
        print(name,spec,metric,flush=True)

@torch.inference_mode()
def evaluate(model,tok,names,split):
    originals={f'model.layers.{i}.mlp.down_proj.weight':layer.mlp.down_proj.weight.detach().cpu().clone()
        for i,layer in enumerate(model.model.layers)}
    questions=e.read(split)
    for name in names:
        path=ROOT/f'results/{split}_{name}.json'
        if path.exists(): raise FileExistsError(path)
        delta_path=ROOT/f'models/{name}_delta.safetensors'
        e.apply(model,originals,load_file(str(delta_path)))
        generated=prior.generate(model,tok,questions,path)
        generated.update(candidate=name,delta_sha256=sha(delta_path),evaluation_role='development')
        e.dump(path,generated)
        print(split,name,prior.summarize(generated['responses']),flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['search','evaluate'])
    p.add_argument('--family',default='strength'); p.add_argument('--candidates',nargs='+')
    p.add_argument('--split',default='validation',choices=['validation','questions'])
    args=p.parse_args()
    for name in ['data','results','reviews','models','runs']: (ROOT/name).mkdir(parents=True,exist_ok=True)
    model,tok=e.load_model()
    if args.stage=='search': search(model,tok,args.family)
    else:
        if not args.candidates: p.error('evaluate requires --candidates')
        evaluate(model,tok,args.candidates,args.split)

if __name__=='__main__': main()
