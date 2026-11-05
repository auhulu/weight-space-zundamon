"""Reproducible, resumable local weight-space experiment; no optimization."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT / 'cache/huggingface'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import argparse
import hashlib
import json
import time
import platform
import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM, set_seed

CFG = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
RUNS = ROOT / 'runs'
RUNS.mkdir(exist_ok=True)

def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def load_model(source=None):
    source = source or CFG['model']
    kw = {'revision': CFG['revision']} if source == CFG['model'] else {}
    model = Qwen3_5ForCausalLM.from_pretrained(source, dtype=torch.float16,
        attn_implementation='sdpa', local_files_only=True, **kw).to('cuda').eval()
    model.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(CFG['model'], revision=CFG['revision'], local_files_only=True)
    return model, tok

def inputs(tok, question, style=False):
    system = CFG['system'] + ('。' + CFG['style'] if style else '')
    messages = [{'role':'system','content':system}, {'role':'user','content':question}]
    return tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_dict=True, return_tensors='pt').to('cuda')

def question(item):
    return item.get('question', item.get('prompt'))

def dataset(split):
    obj = json.loads((ROOT / 'data' / f'{split}.json').read_text(encoding='utf-8'))
    return obj if isinstance(obj, list) else obj['items']

@torch.inference_mode()
def generate(model, tok, q, seed, style=False, greedy=False):
    set_seed(seed)
    x = inputs(tok, q, style)
    start = time.perf_counter()
    args = dict(max_new_tokens=CFG['max_new_tokens'], do_sample=not greedy,
        pad_token_id=tok.eos_token_id, use_cache=True)
    if not greedy:
        args.update(temperature=CFG['temperature'],top_p=CFG['top_p'],top_k=CFG['top_k'])
    out = model.generate(**x, **args)[0, x['input_ids'].shape[1]:]
    text = tok.decode(out, skip_special_tokens=True)
    return {'text':text, 'tokens':len(out), 'seconds':time.perf_counter()-start,
        'hit_limit':len(out) == CFG['max_new_tokens'], 'token_ids':out.cpu().tolist()}

def evaluate(model, tok, split, condition, seeds, limit=None):
    path = RUNS / f'{split}_{condition}.jsonl'
    old = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    done = {(x['id'],x['seed']) for x in old}
    rows = dataset(split)[:limit]
    for seed in seeds:
        for item in rows:
            if (item['id'], seed) in done:
                continue
            result = generate(model,tok,question(item),seed,condition=='prompt')
            row = {**item, **result, 'condition':condition,'seed':seed}
            with path.open('a',encoding='utf-8') as f:
                f.write(json.dumps(row,ensure_ascii=False)+'\n')
            print(json.dumps({'split':split,'condition':condition,'id':item['id'],
                'seed':seed,'seconds':round(result['seconds'],2),'tokens':result['tokens']},ensure_ascii=False),flush=True)
    return path

@torch.inference_mode()
def collect(model, tok):
    stats_path = RUNS / 'train_stats.pt'
    if stats_path.exists():
        return
    records = [[] for _ in model.model.layers]
    for item in dataset('train'):
        paired = []
        for style in [False,True]:
            current = [{} for _ in records]
            handles=[]
            for i, layer in enumerate(model.model.layers):
                def pre(module,args,i=i):
                    current[i]['h'] = args[0][0,-1].float().cpu()
                def post(module,args,out,i=i):
                    if isinstance(out,tuple): out=out[0]
                    current[i]['residual'] = out[0,-1].float().cpu()
                handles += [layer.mlp.down_proj.register_forward_pre_hook(pre),layer.register_forward_hook(post)]
            try:
                model(**inputs(tok,question(item),style),use_cache=False,logits_to_keep=1)
            finally:
                for h in handles: h.remove()
            paired.append(current)
        for i in range(len(records)):
            records[i].append({'h':paired[0][i]['h'], 'residual':paired[0][i]['residual'],
                'd':paired[1][i]['residual']-paired[0][i]['residual']})
        print('collected '+item['id'],flush=True)
    layers=[]
    for i, rec in enumerate(records):
        h=torch.stack([r['h'] for r in rec]); d=torch.stack([r['d'] for r in rec]); r=torch.stack([r['residual'] for r in rec])
        layers.append({'layer':i,'module_name':f'model.layers.{i}.mlp.down_proj',
            'h_mean':h.mean(0),'h_second_moment':h.square().mean(0),
            'd_mean':d.mean(0),'d_rms':d.square().sum(1).mean().sqrt().item(),
            'residual_rms':r.square().sum(1).mean().sqrt().item()})
    torch.save({'layers':layers,'train_count':len(dataset('train'))},stats_path)
    dump(RUNS/'stats_summary.json',[{k:(v.norm().item() if torch.is_tensor(v) else v) for k,v in row.items()} for row in layers])

def read_rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]

def tensor_digest(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage', choices=['smoke','prepare','candidates','final'])
    args=parser.parse_args()
    if args.stage!='smoke':
        import freeze_data  # Verify the fixed dataset/config/protocol manifest.
    torch.set_num_threads(6)
    model,tok=load_model()
    if args.stage=='smoke':
        out=generate(model,tok,'虹はどうしてできるのですか？',42)
        dump(RUNS/'smoke.json',out)
        print(json.dumps(out,ensure_ascii=False),flush=True)
        dump(RUNS/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,
            'gpu':torch.cuda.get_device_name(),'cuda':torch.version.cuda,'config':CFG,
            'parameters':sum(p.numel() for p in model.parameters()),'dtype':str(next(model.parameters()).dtype)})
    elif args.stage=='prepare':
        collect(model,tok)
        for split in ['train','validation']:
            for cond in ['base','prompt']:
                evaluate(model,tok,split,cond,[CFG['selection_seed']])
    elif args.stage=='candidates':
        from edit_design import design_candidates, apply_edit
        stats=torch.load(RUNS/'train_stats.pt',weights_only=True)
        specs=design_candidates(stats)
        dump(RUNS/'candidate_specs.json',specs)
        originals={f'model.layers.{i}.mlp.down_proj':l.mlp.down_proj.weight.detach().cpu().clone() for i,l in enumerate(model.model.layers)}
        for spec in specs:
            with torch.no_grad():
                for name,w in originals.items(): model.get_submodule(name).weight.copy_(w)
            metadata=apply_edit(model,stats,spec)
            dump(RUNS/f"edit_{spec['id']}.json",metadata)
            evaluate(model,tok,'validation',spec['id'],[CFG['selection_seed']])
        with torch.no_grad():
            for name,w in originals.items(): model.get_submodule(name).weight.copy_(w)
        assert all(torch.equal(model.get_submodule(n).weight.cpu(),w) for n,w in originals.items())
        print('candidates complete; select using select.py',flush=True)
    elif args.stage=='final':
        from edit_design import apply_edit
        selection=json.loads((RUNS/'selection.json').read_text(encoding='utf-8'))
        for split in ['test','side']:
            for condition in ['base','prompt']:
                evaluate(model,tok,split,condition,CFG['final_seeds'])
        stats=torch.load(RUNS/'train_stats.pt',weights_only=True)
        before={n:tensor_digest(p) for n,p in model.named_parameters()}
        edited_names=[e['module_name']+'.weight' for e in selection['spec']['edits']]
        original_tensors={n:p.detach().cpu().clone() for n,p in model.named_parameters() if n in edited_names}
        metadata=apply_edit(model,stats,selection['spec'])
        after={n:tensor_digest(p) for n,p in model.named_parameters()}
        changed=[n for n in before if before[n]!=after[n]]
        assert changed and all(n.endswith('.mlp.down_proj.weight') for n in changed)
        assert all(torch.isfinite(p).all().item() for p in model.parameters())
        check_q=question(dataset('train')[0])
        expected=generate(model,tok,check_q,42,greedy=True)
        saved=ROOT/'models'/'edited'
        from safetensors.torch import save_file
        saved.parent.mkdir(parents=True,exist_ok=True)
        save_file({n:(model.get_parameter(n).detach().cpu().float()-original_tensors[n].float())
            for n in edited_names},str(saved.parent/'edit_delta.safetensors'))
        model.save_pretrained(saved,safe_serialization=True)
        del model
        torch.cuda.empty_cache()
        model,tok=load_model(str(saved))
        actual=generate(model,tok,check_q,42,greedy=True)
        assert actual['token_ids']==expected['token_ids'], 'reload changed greedy outputs'
        reloaded={n:tensor_digest(p) for n,p in model.named_parameters()}
        assert reloaded==after, 'reload changed tensors'
        dump(RUNS/'verification.json',{'changed_parameters':changed,'unchanged_parameters':len(before)-len(changed),
            'finite':True,'reload_tensors_identical':True,'reload_greedy_identical':True,'edit':metadata})
        for split in ['test','side']:
            evaluate(model,tok,split,'edited',CFG['final_seeds'])

if __name__=='__main__': main()
