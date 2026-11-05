"""Export complete seed blocks, each balanced across all three conditions."""
import hashlib
import json
import random
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def export():
    blind=ROOT/'runs'/'blind';blind.mkdir(exist_ok=True)
    mapping={}
    rows={c:[json.loads(s) for s in (ROOT/'runs'/f'test_{c}.jsonl').read_text(encoding='utf-8').splitlines()]
        if (ROOT/'runs'/f'test_{c}.jsonl').exists() else [] for c in ['base','prompt','edited']}
    ready=[]
    for i,seed in enumerate([101,202,303]):
        selected={c:[r for r in rs if r['seed']==seed] for c,rs in rows.items()}
        if not all(len(rs)==48 for rs in selected.values()):continue
        packet=[]
        for c,rs in selected.items():
            for r in rs:
                rid=hashlib.sha256(f"blind2026:{c}:{r['id']}:{r['seed']}".encode()).hexdigest()[:12]
                packet.append({'review_id':rid,'question':r.get('question',r.get('prompt')),'answer':r['text']})
                mapping[rid]={'condition':c,'id':r['id'],'seed':r['seed']}
        random.Random(8931+seed).shuffle(packet)
        path=blind/f'packet_{i}.json'
        if path.exists():assert json.loads(path.read_text(encoding='utf-8'))==packet
        else:path.write_text(json.dumps(packet,ensure_ascii=False,indent=2),encoding='utf-8')
        ready.append(i)
    (ROOT/'runs'/'blind_mapping.json').write_text(json.dumps(mapping,indent=2),encoding='utf-8')
    print('Ready blind packets:',ready)
    return ready
if __name__=='__main__':export()
