"""Controlled continuation: only data loss weight differs between the two arms."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from channel_operator.model import load_model,residuals,outlet_residuals
from channel_operator.profiles import sample_profiles
from channel_operator.v2_train import monitoring_errors


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--data',required=True)
    p.add_argument('--monitor',required=True);p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=500)
    a=p.parse_args();torch.set_num_threads(4)
    with np.load(a.data) as d:data={k:d[k].copy() for k in d.files}
    with np.load(a.monitor) as d:monitor={k:d[k].copy() for k in d.files}
    tensor=lambda x:torch.tensor(x,dtype=torch.float32)
    td={k:tensor(data[k]) for k in ['profiles','reynolds','xy','fields']}
    probe_rng=np.random.default_rng(917)
    probe=tensor(probe_rng.uniform([.005,.005],[4.995,.995],(4,512,2)))
    probe_ids=[12,45,66,91]
    results={'controlled_difference':'data_weight only (4000 versus 20)','steps':a.steps,'arms':{}}
    for weight in [4000.,20.]:
        model,state=load_model(a.checkpoint)
        optimizer=torch.optim.Adam(model.parameters(),lr=2e-4)
        optimizer.load_state_dict(state['optimizer'])
        for group in optimizer.param_groups:group['lr']=2e-4
        torch.manual_seed(1913);rng=np.random.default_rng(1913)
        def measure():
            mass,mx,my=residuals(model,td['profiles'][probe_ids],td['reynolds'][probe_ids],probe)
            return {'training_condition_fresh_points_momentum_rms':float((mx.square().mean()+my.square().mean()).sqrt().detach()),
                    'training_condition_fresh_points_divergence_rms':float(mass.square().mean().sqrt().detach()),
                    'validation':monitoring_errors(model,monitor,tensor)}
        before=measure()
        for step in range(a.steps):
            model.train();optimizer.zero_grad(set_to_none=True)
            profile,_=sample_profiles(rng,8);re=np.exp(rng.uniform(np.log(10),np.log(100),8))
            xy=torch.rand(8,64,2);xy[...,0]=5*xy[...,0]**2
            mass,mx,my=residuals(model,tensor(profile),tensor(re),xy)
            out=torch.rand(8,32,2);out[...,0]=5
            on,ot=outlet_residuals(model,tensor(profile),tensor(re),out)
            ci=torch.randint(len(td['profiles']),(8,));pi=torch.randint(len(td['xy']),(192,))
            pred=model(td['profiles'][ci],td['reynolds'][ci],td['xy'][pi][None].expand(8,-1,-1))
            target=td['fields'][ci[:,None],pi[None,:]]
            scale=torch.ones(8,1,3);scale[...,2]=td['reynolds'][ci,None]/60
            supervised=((pred-target)*scale).square().mean()
            gx=torch.rand(8,3,1)*5;gy=torch.linspace(0,1,49)
            grid=torch.stack([gx.expand(-1,-1,49),gy[None,None,:].expand(8,3,-1)],-1).reshape(8,-1,2)
            u=model(tensor(profile),tensor(re),grid)[...,0].reshape(8,3,49)
            flux=(torch.trapezoid(u,gy,dim=-1)-1).square().mean()
            loss=10*mass.square().mean()+mx.square().mean()+my.square().mean()+.1*(on.square().mean()+ot.square().mean())+10*flux+weight*supervised
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        after=measure()
        results['arms'][str(int(weight))]={'before':before,'after':after}
        print(json.dumps({'data_weight':weight,'before_momentum':before['training_condition_fresh_points_momentum_rms'],
                          'after_momentum':after['training_condition_fresh_points_momentum_rms'],
                          'validation':after['validation']}),flush=True)
    Path(a.output).write_text(json.dumps(results,indent=2))


if __name__=='__main__':main()
