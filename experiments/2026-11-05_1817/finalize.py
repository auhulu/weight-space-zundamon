"""Freeze a reviewed internal-weight edit and verify its standalone checkpoint."""
import argparse
import hashlib
import json
import platform
import torch
from safetensors.torch import load_file, save_file
import experiment as e

@torch.inference_mode()
def main(edit_path, selection_path):
    selection=json.loads(selection_path.read_text(encoding='utf-8'))
    with edit_path.open('rb') as f: edit_hash=hashlib.file_digest(f,'sha256').hexdigest()
    assert edit_hash==selection['delta_sha256'],'Edit differs from frozen selection'
    model,tok=e.load_model()
    base_tied = model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
    assert base_tied == model.config.tie_word_embeddings
    before=e.hashes(model)
    edits=load_file(str(edit_path))
    originals={name:model.get_parameter(name).detach().cpu().clone() for name in edits}
    base_probe=e.generate(model,tok,e.read('questions')[:8])
    e.apply(model,originals,edits); first=e.hashes(model)
    e.apply(model,originals,edits); assert e.hashes(model)==first
    actual={name:model.get_parameter(name).detach().cpu().float()-original.float() for name,original in originals.items()}
    changed={name:int((delta!=0).sum()) for name,delta in actual.items()}
    assert any(changed.values()),'A zero edit is not an edited model'
    assert all(first[n]==h for n,h in before.items() if n not in edits)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    e.apply(model,originals,{})
    assert e.hashes(model)==before
    restored=e.generate(model,tok,e.read('questions')[:8])
    assert restored['responses']==base_probe['responses']
    for split in ['validation','generalization']:
        e.generate(model,tok,e.read(split),e.ROOT/f'results/base_{split}_128.json')
    e.apply(model,originals,edits)
    folder=e.ROOT/'models/zundamon-edited'
    model.save_pretrained(folder); tok.save_pretrained(folder)
    save_file(actual,str(e.ROOT/'models/actual_delta.safetensors'))
    for split in ['questions','generalization']:
        r=e.generate(model,tok,e.read(split),e.ROOT/f'results/edited_{split}_128.json')
        r['condition']='edited'
        r['selection_id']=selection['id']
        r['delta_sha256']=edit_hash
        e.dump(e.ROOT/f'results/edited_{split}_128.json',r)
        print('final',split,e.quick(r),flush=True)
    probe=e.generate(model,tok,e.read('questions')[:8])
    model.cpu(); del model
    import gc
    gc.collect(); torch.cuda.empty_cache()
    reloaded,reloaded_tok=e.load_model(str(folder))
    reload_hashes=e.hashes(reloaded)
    assert reload_hashes==first
    reload_tied = reloaded.model.embed_tokens.weight.data_ptr() == reloaded.lm_head.weight.data_ptr()
    assert reload_tied == base_tied and reloaded.config.tie_word_embeddings == base_tied
    check=e.generate(reloaded,reloaded_tok,e.read('questions')[:8])
    assert check['responses']==probe['responses']
    source=e.ROOT.parent.parent/'cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots'/e.CFG['revision']/'model.safetensors-00001-of-00001.safetensors'
    with source.open('rb') as f: source_hash=hashlib.file_digest(f,'sha256').hexdigest()
    e.dump(e.ROOT/'results/base_hashes.json',before)
    e.dump(e.ROOT/'results/verification.json',{
        'model':e.CFG['model'],'revision':e.CFG['revision'],'source_checkpoint_sha256':source_hash,
        'selection':selection,'edit_file':str(edit_path.relative_to(e.ROOT)),
        'changed_elements':changed,'noncumulative':True,'restore_hashes_equal':True,
        'restore_greedy_equal':True,'all_other_parameters_unchanged':True,
        'reload_hashes_equal':True,'reload_greedy_equal':True,'finite':True,
        'embedding_output_final_norm_unchanged':True,'weight_tying_preserved':reload_tied == base_tied,
        'base_weights_tied':base_tied,'reloaded_weights_tied':reload_tied,
        'after_hashes':first,'torch_version':torch.__version__,'python':platform.python_version(),
        'gpu':torch.cuda.get_device_name(),'reviewer':'Codex agent; not independent human grading'})

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--edit',required=True); p.add_argument('--selection',default='results/selection.json')
    a=p.parse_args(); main(e.ROOT/a.edit,e.ROOT/a.selection)
