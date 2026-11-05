"""Rebuild the frozen edit without repeating candidate selection."""
import hashlib
import json
import torch
from safetensors.torch import save_file
import experiment as e
from contrast import delta_for

def main():
    choice=json.loads((e.ROOT/'results/selection.json').read_text(encoding='utf-8'))
    spec=choice['spec']
    stats_path='runs/prefix_stats_post.pt' if spec['method']=='terminal_gated' else 'runs/prefix_stats.pt'
    stats=torch.load(e.ROOT/stats_path,weights_only=True)
    model,_=e.load_model()
    base={i:model.model.layers[i].mlp.down_proj.weight.detach().cpu().clone() for i in spec['layers']}
    edits=delta_for(stats,base,spec)
    path=e.ROOT/f'models/{choice["id"]}_delta.safetensors'
    path.parent.mkdir(exist_ok=True)
    temporary=path.with_suffix('.rebuild.safetensors')
    save_file(edits,str(temporary))
    with temporary.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
    assert actual==choice['delta_sha256'],'Rebuilt edit differs from the frozen artifact'
    temporary.replace(path)
    print('Frozen edit reproduced:',choice['id'],actual)

if __name__=='__main__': main()
