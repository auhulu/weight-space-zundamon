"""Algebraic and experimental-boundary checks without a GPU/model load."""
import json
from types import SimpleNamespace
import torch
import round2 as r

def test_disjoint_questions():
    splits=[r.data(s) for s in ['train','validation','test']]
    assert list(map(len,splits))==[24,24,48]
    questions=[x['prompt'] for split in splits for x in split]
    assert len(questions)==len(set(questions))
    old={x['prompt'] for s in ['train','validation','test'] for x in r.e.dataset(s)}
    assert not old.intersection(questions)

def test_sparse_write_calibration_and_random_control():
    torch.manual_seed(11)
    layers=[SimpleNamespace(mlp=SimpleNamespace(down_proj=SimpleNamespace(weight=torch.randn(8,12).half()))) for _ in range(20)]
    model=SimpleNamespace(model=SimpleNamespace(layers=layers)); base=r.originals(model)
    stats={}
    for i in r.LAYERS:
        d=torch.randn(8); u=d/d.norm(); key=torch.randn(12); key[3:]=0
        random=torch.randn(8); random-=u*(u@random); random/=random.norm()
        stats[i]=dict(d=d,u=u,key=key,mask=(key!=0).float(),random=random)
    spec=dict(layers=r.LAYERS,alpha=2.,method='write')
    edit=r.apply(model,base,stats,spec)
    for i in r.LAYERS:
        delta=layers[i].mlp.down_proj.weight.float()-base[i].float(); s=stats[i]
        assert torch.equal(layers[i].mlp.down_proj.weight[:,3:],base[i][:,3:])
        assert torch.allclose(delta@s['key'],2*s['d'],atol=.004,rtol=.004)
    control=r.apply(model,base,stats,{**spec,'random':True})
    for actual,random in zip(edit['layers'],control['layers']):
        assert abs(actual['planned_norm']-random['planned_norm'])<1e-5
    r.apply(model,base,stats,{**spec,'method':'project'})
    for i in r.LAYERS:
        u=stats[i]['u']; old_projection=u@base[i].float()
        projected=u@layers[i].mlp.down_proj.weight.float()
        assert torch.allclose(projected[:3],3*old_projection[:3],atol=.004,rtol=.004)
        assert torch.equal(layers[i].mlp.down_proj.weight[:,3:],base[i][:,3:])
    r.restore(model,base)
    assert all(torch.equal(layers[i].mlp.down_proj.weight,base[i]) for i in r.LAYERS)

def test_predeclared_grid():
    specs=r.specs()
    assert len(specs)==13
    assert sum(not s.get('random',False) for s in specs)==10
    assert len({s['id'] for s in specs})==13
