"""Reconstruct the frozen geometric delta, without model generation or fitting."""
import json
import run as r
import sys
sys.path.insert(0,str(r.ROOT))
import torch
from safetensors.torch import save_file
from finalize import sha256, local_path


def main():
    selection=json.loads((r.ROOT/'results/selection.json').read_text(encoding='utf-8'))
    for name in ['run.py','data/pairs.json']:
        if sha256(r.ROOT/name,text=True)!=selection['file_hashes'][name]:
            raise ValueError(f'Frozen source changed: {name}')
    stats=torch.load(r.ROOT/'runs/fewshot_stats.pt',map_location='cpu',weights_only=True)
    edits,_=r.make_edit(stats,selection['spec'])
    folder=r.ROOT/'runs/rebuild'; folder.mkdir(parents=True,exist_ok=True)
    temporary=folder/'rebuilt_delta.safetensors'; save_file(edits,str(temporary))
    if sha256(temporary)!=selection['delta_sha256']:
        raise ValueError('Reconstructed delta differs from the frozen result')
    destination=local_path(selection['edit_file'],'models'); destination.parent.mkdir(exist_ok=True)
    if destination.exists() and sha256(destination)!=selection['delta_sha256']:
        raise ValueError('Existing delta differs; refusing to overwrite it')
    if not destination.exists(): destination.write_bytes(temporary.read_bytes())
    print(json.dumps({'delta':str(destination),'sha256':sha256(destination),'match':True},indent=2))


if __name__ == '__main__': main()
