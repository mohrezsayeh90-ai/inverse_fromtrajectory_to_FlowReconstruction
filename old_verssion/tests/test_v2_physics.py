import numpy as np
import torch
from channel_operator.v2_model import StreamDeepONet,physics
from channel_operator.v2_profiles import inlet_values,sample,monitoring_conditions


def test_stream_model_mass_inlet_and_walls():
    torch.manual_seed(8);model=StreamDeepONet(16,8,2).double()
    p=torch.tensor(inlet_values(np.array([[.1,-.2,.1,0],[0,0,0,.6]])))
    re=torch.tensor([10.,100.],dtype=torch.float64)
    y=torch.linspace(0,1,33,dtype=torch.float64).repeat(2,1)
    xy=torch.stack([torch.zeros_like(y),y],-1)
    uvp=model(p,re,xy)
    assert (uvp[...,0]-p).abs().max()<1e-6
    assert uvp[...,1].abs().max()<1e-7
    for wall in [0.,1.]:
        xy=torch.stack([5*y,torch.full_like(y,wall)],-1)
        assert model(p,re,xy)[...,:2].abs().max()<1e-6
    xy=torch.rand(2,24,2,dtype=torch.float64);xy[...,0]*=5
    mx,my,mass=physics(model,p,re,xy)
    assert mass.abs().max()<1e-10
    (mx.square().mean()+my.square().mean()).backward()
    assert model.branch.output.weight.grad.abs().sum()>0


def test_v2_poiseuille_equations_and_no_grad_inference():
    model=StreamDeepONet(16,8,2).double()
    with torch.no_grad():model.trunk.output.weight.zero_();model.trunk.output.bias.zero_()
    p=torch.tensor(inlet_values(np.zeros((1,4))))
    re=torch.tensor([40.],dtype=torch.float64)
    xy=torch.rand(1,32,2,dtype=torch.float64);xy[...,0]*=5
    for r in physics(model,p,re,xy): assert r.abs().max()<1e-4
    with torch.no_grad():fields=model(p,re,xy)
    assert not fields.requires_grad
    assert np.isfinite(fields.numpy()).all()


def test_training_sampler_excludes_exact_monitor_inlets():
    _,_,train=sample(np.random.default_rng(42),1000)
    _,_,monitor=monitoring_conditions()
    assert np.abs(train).sum(1).max()<=.8000001
    assert np.min(np.abs(train[:,None,:]-monitor[None,:,:]).max(-1))>1e-8
