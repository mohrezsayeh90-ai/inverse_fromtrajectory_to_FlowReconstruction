"""Divergence-free physics-informed DeepONet with paper's modified tanh MLPs."""
import torch
from torch import nn
from .v2_profiles import PROJECT, INTEGRALS
from .profiles import Y
from .model import gradient


class ModifiedMLP(nn.Module):
    def __init__(self, inputs, width, outputs, depth=3):
        super().__init__()
        self.u=nn.Linear(inputs,width); self.v=nn.Linear(inputs,width)
        self.gates=nn.ModuleList([nn.Linear(inputs,width)]+[nn.Linear(width,width) for _ in range(depth-1)])
        self.output=nn.Linear(width,outputs)
        for layer in self.modules():
            if isinstance(layer,nn.Linear):
                nn.init.xavier_normal_(layer.weight);nn.init.zeros_(layer.bias)

    def forward(self,x):
        u,v=torch.tanh(self.u(x)),torch.tanh(self.v(x))
        h=x
        for gate in self.gates:
            z=torch.tanh(gate(h));h=z*u+(1-z)*v
        return self.output(h)


class StreamDeepONet(nn.Module):
    def __init__(self,width=64,latent=32,depth=3,pressure_mode='viscous'):
        super().__init__()
        self.config=dict(width=width,latent=latent,depth=depth)
        self.pressure_mode=pressure_mode
        if pressure_mode not in ['viscous','convective']:raise ValueError('Unknown pressure mode')
        if pressure_mode!='viscous':self.config['pressure_mode']=pressure_mode
        self.latent=latent
        self.branch=ModifiedMLP(34,width,2*latent,depth)
        self.trunk=ModifiedMLP(14,width,2*latent,depth)
        nn.init.normal_(self.trunk.output.weight,std=1e-3)
        self.register_buffer('projection',torch.tensor(PROJECT,dtype=torch.float32))
        self.register_buffer('integrals',torch.tensor(INTEGRALS,dtype=torch.float32))
        self.register_buffer('base',torch.tensor(6*Y*(1-Y),dtype=torch.float32))

    def inlet_parameters(self,profiles):
        b=profiles@self.projection.T
        flux=b[:,0]+(b[:,1:]*self.integrals).sum(-1)
        return b/flux[:,None]

    def potentials(self,profiles,re,xy):
        x,y=xy[...,0],xy[...,1]
        condition=torch.cat([profiles-self.base,torch.log(re/10)[:,None]/torch.log(re.new_tensor(10.))],-1)
        branch=self.branch(condition).reshape(-1,2,self.latent)
        modes=torch.arange(1,5,dtype=xy.dtype,device=xy.device)
        alpha=torch.pi*modes
        phase=y[...,None]*alpha
        features=torch.cat([torch.stack([2*x/5-1,2*y-1],-1),torch.sin(phase),torch.cos(phase),
                            torch.stack([torch.log1p(20*x)/torch.log(x.new_tensor(101.)),torch.exp(-x),torch.exp(-4*x),torch.exp(-16*x)],-1)],-1)
        trunk=self.trunk(features).reshape(*x.shape,2,self.latent)
        raw=torch.einsum('bol,bnol->bno',branch,trunk)/self.latent**.5
        b=self.inlet_parameters(profiles)
        base=3*y*y-2*y**3
        # Analytic integrals of 6 y(1-y) cos(k*pi*y), zero at y=0.
        ik=6*((y-y*y)[...,None]*torch.sin(phase)/alpha + (1-2*y)[...,None]*torch.cos(phase)/alpha**2
              +2*torch.sin(phase)/alpha**3-1/alpha**2)
        deviation=ik-base[...,None]*self.integrals
        # Mode-dependent viscous development scale, corrected by the learned field.
        ell=.1+.05*re[:,None,None]/modes**2
        decay=torch.exp(-(x[...,None]/ell)**2)
        psi=base+(b[:,None,1:]*deviation*decay).sum(-1)
        mask=16*y*y*(1-y)**2*(1-torch.exp(-x/.2))**2
        psi=psi+mask*raw[...,0]
        pressure_scale=12/re[:,None] if self.pressure_mode=='viscous' else 1.
        p=12/re[:,None]*(5-x)+pressure_scale*raw[...,1]
        return psi,p

    def forward(self,profiles,re,xy):
        active=torch.is_grad_enabled()
        with torch.enable_grad():
            if not xy.requires_grad:
                xy=xy.detach().requires_grad_(True)
            psi,p=self.potentials(profiles,re,xy)
            g=torch.autograd.grad(psi.sum(),xy,create_graph=active,retain_graph=True)[0]
            fields=torch.stack([g[...,1],-g[...,0],p],-1)
        return fields if active else fields.detach()


def physics(model,profiles,re,xy):
    xy=xy.detach().requires_grad_(True)
    fields=model(profiles,re,xy)
    u,v,p=fields.unbind(-1)
    gu,gv,gp=[gradient(z,xy) for z in (u,v,p)]
    lu=gradient(gu[...,0],xy)[...,0]+gradient(gu[...,1],xy)[...,1]
    lv=gradient(gv[...,0],xy)[...,0]+gradient(gv[...,1],xy)[...,1]
    mx=u*gu[...,0]+v*gu[...,1]+gp[...,0]-lu/re[:,None]
    my=u*gv[...,0]+v*gv[...,1]+gp[...,1]-lv/re[:,None]
    return mx,my,gu[...,0]+gv[...,1]


def outlet(model,profiles,re,y):
    xy=torch.stack([torch.full_like(y,5.),y],-1).requires_grad_(True)
    fields=model(profiles,re,xy)
    ux=gradient(fields[...,0],xy)[...,0]
    vx=gradient(fields[...,1],xy)[...,0]
    return ux-re[:,None]*fields[...,2],vx


def load(path,device='cpu'):
    state=torch.load(path,map_location=device,weights_only=False)
    model=StreamDeepONet(**state['config']).to(device)
    model.load_state_dict(state['state_dict']);model.eval()
    return model,state
