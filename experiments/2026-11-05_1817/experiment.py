"""Independent 1817 experiment. All weight updates are gradient-free."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT.parent.parent / 'cache/huggingface'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM
from safetensors.torch import save_file

CFG = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
torch.set_num_threads(4)

def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8', newline='\n')

def read(name):
    return json.loads((ROOT / 'data' / (name + '.json')).read_text(encoding='utf-8'))

def digest(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def hashes(model):
    return {n: digest(p) for n, p in model.named_parameters()}

def load_model(source=None):
    source = source or CFG['model']
    kw = {'revision': CFG['revision']} if source == CFG['model'] else {}
    model = Qwen3_5ForCausalLM.from_pretrained(source, dtype=torch.float16,
        attn_implementation='sdpa', local_files_only=True, **kw).to('cuda').eval()
    model.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(source, local_files_only=True, **kw)
    tok.padding_side = 'left'
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    return model, tok

def inputs(tok, questions):
    messages = [[{'role':'system','content':CFG['system']}, {'role':'user','content':q}] for q in questions]
    return tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
        enable_thinking=False, padding=True, return_dict=True, return_tensors='pt').to('cuda')

@torch.inference_mode()
def generate(model, tok, questions, path=None):
    started = perf_counter(); responses = []
    eos = model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos)
    for offset in range(0, len(questions), CFG['batch_size']):
        batch = questions[offset:offset+CFG['batch_size']]
        x = inputs(tok, [r['prompt'] for r in batch])
        y = model.generate(**x, max_new_tokens=128, do_sample=False,
            pad_token_id=tok.pad_token_id, use_cache=True)[:, x['input_ids'].shape[1]:]
        for row, tokens in zip(batch, y.tolist(), strict=True):
            end = next((i+1 for i,t in enumerate(tokens) if t in eos), len(tokens))
            tokens = tokens[:end]
            responses.append({**row, 'answer': tok.decode(tokens, skip_special_tokens=True).strip(),
                'tokens': len(tokens), 'hit_limit':len(tokens)==128 and eos.isdisjoint(tokens)})
    result = {**CFG, 'elapsed_seconds':round(perf_counter()-started,2), 'responses':responses}
    if path: dump(path, result)
    return result

def quick(result):
    # Screening only; final acceptance requires per-answer semantic review.
    import re
    rr = result['responses']
    good = sum(bool(re.search(r'(?:なのだ|のだ)[。！？!?]?(?:\s|$)', r['answer'])) and len(r['answer'])>6 for r in rr)
    bad = sum(not r['answer'] or r['hit_limit'] or bool(re.search(r'(.{5,})\1\1',r['answer'])) for r in rr)
    return {'style_proxy':good/len(rr),'bad':bad,'n':len(rr)}

@torch.inference_mode()
def collect(model, tok):
    records = {i:[] for i in range(len(model.model.layers))}
    for number, row in enumerate(read('train_pairs')):
        paired = []
        for field in ['normal','style']:
            prefix = inputs(tok,[row['prompt']])['input_ids']
            answer = tok(row[field],add_special_tokens=False,return_tensors='pt')['input_ids'].to('cuda')
            x = torch.cat([prefix,answer],1); width=min(16,answer.shape[1]); current={i:{} for i in records}; handles=[]
            for i, layer in enumerate(model.model.layers):
                def pre(m,args,i=i): current[i]['g']=args[0][0,-width:].float().mean(0).cpu()
                def post(m,args,out,i=i): current[i]['h']=(out[0] if isinstance(out,tuple) else out)[0,-width:].float().mean(0).cpu()
                handles.extend([layer.mlp.down_proj.register_forward_pre_hook(pre),layer.register_forward_hook(post)])
            try: model.model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False)
            finally:
                for h in handles: h.remove()
            paired.append(current)
        for i in records:
            records[i].append({'g':paired[0][i]['g'], 'dg':paired[1][i]['g']-paired[0][i]['g'], 'd':paired[1][i]['h']-paired[0][i]['h']})
        if (number+1)%10==0: print('collect',number+1,flush=True)
    stats={}
    for i, rr in records.items():
        g=torch.stack([r['g'] for r in rr]); d=torch.stack([r['d'] for r in rr]); dg=torch.stack([r['dg'] for r in rr]).mean(0)
        u=d.mean(0); coherence=float(u.norm()/d.norm(dim=1).square().mean().sqrt()); u=u/u.norm()
        w=model.model.layers[i].mlp.down_proj.weight.float().cpu()
        stats[i]={'g':g,'d':d,'u':u,'writer':dg*(u@w),'coherence':coherence}
    (ROOT/'runs').mkdir(exist_ok=True)
    torch.save(stats,ROOT/'runs/activation_stats.pt')
    dump(ROOT/'results/activation_summary.json',{i:{'coherence':s['coherence'],'d_norm':float(s['d'].mean(0).norm())} for i,s in stats.items()})
    return stats

def mlp_delta(stats, base, spec):
    edits={}
    for i in spec['layers']:
        s=stats[i]; w=base[i].float(); mask=torch.zeros(w.shape[1]); mask[s['writer'].topk(spec['columns']).indices]=1
        if spec['method']=='project': delta=torch.outer(s['u'],mask*(s['u']@w))
        elif spec['method']=='write':
            key=s['g'].mean(0)*mask
            delta=torch.outer(s['d'].mean(0),key/key.square().sum().clamp_min(1e-8))
        else:
            raise ValueError('Only hypothesis-driven write/project edits are permitted')
        edits[f'model.layers.{i}.mlp.down_proj.weight']=delta*spec['alpha']/len(spec['layers'])
    return edits

@torch.no_grad()
def apply(model, originals, edits):
    import re
    for name in set(originals) | set(edits):
        if not re.fullmatch(r'model\.layers\.\d+\.mlp\.down_proj\.weight',name):
            raise ValueError('Only internal MLP down projections may be edited')
    prepared = {}
    for name, delta in edits.items():
        p = model.get_parameter(name)
        if name not in originals or delta.shape != p.shape or originals[name].shape != p.shape:
            raise ValueError('Edit shape and original must match the target parameter')
        candidate = (originals[name].float() + delta).to(device=p.device, dtype=p.dtype)
        if not torch.isfinite(candidate).all():
            raise ValueError('Nonfinite edited parameter')
        prepared[name] = candidate
    for name, original in originals.items():
        p = model.get_parameter(name)
        p.copy_(original.to(device=p.device, dtype=p.dtype))
    for name, candidate in prepared.items():
        model.get_parameter(name).copy_(candidate)

@torch.inference_mode()
def search_mlp(model,tok):
    stats=torch.load(ROOT/'runs/activation_stats.pt',weights_only=True)
    ranked=sorted((i for i in stats if i>=6),key=lambda i:-stats[i]['coherence'])[:4]
    base={i:model.model.layers[i].mlp.down_proj.weight.detach().cpu().clone() for i in stats}
    originals={f'model.layers.{i}.mlp.down_proj.weight':w for i,w in base.items()}
    val=read('validation'); screen=val[::5]; scores=[]
    for method in ['write','project']:
        for layers in [[i] for i in ranked]+[ranked[:2]]:
            for alpha in [.5,2.,6.]:
                spec={'method':method,'layers':layers,'alpha':alpha,'columns':64}
                name=f"{method}_{'-'.join(map(str,layers))}_{alpha}"
                edits=mlp_delta(stats,base,spec); apply(model,originals,edits)
                result=generate(model,tok,screen,ROOT/f'runs/mlp_{name}.json'); metric=quick(result)
                scores.append({'id':name,'spec':spec,**metric}); print(name,metric,flush=True)
    top=sorted(scores,key=lambda s:(-s['style_proxy'],s['bad']))[:3]
    for s in top:
        edits=mlp_delta(stats,base,s['spec']); apply(model,originals,edits)
        r=generate(model,tok,val,ROOT/f'results/validation_{s["id"]}.json'); s['full']=quick(r)
        save_file(edits,str(ROOT/f'models/{s["id"]}_delta.safetensors'))
    apply(model,originals,{})
    dump(ROOT/'results/mlp_search.json',{'ranked_layers':ranked,'screen':scores,'finalists':top})

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['baseline','collect','mlp','generate'])
    parser.add_argument('--model'); parser.add_argument('--data',default='questions'); parser.add_argument('--output')
    args=parser.parse_args(); (ROOT/'models').mkdir(exist_ok=True)
    model,tok=load_model(args.model)
    if args.stage=='baseline':
        dump(ROOT/'results/base_hashes.json',hashes(model))
        for split in ['questions','validation','generalization']:
            result=generate(model,tok,read(split),ROOT/f'results/base_{split}_128.json'); print(split,quick(result),flush=True)
    elif args.stage=='collect': collect(model,tok)
    elif args.stage=='mlp': search_mlp(model,tok)
    else: generate(model,tok,read(args.data),ROOT/(args.output or 'results/generated.json'))

if __name__=='__main__': main()
