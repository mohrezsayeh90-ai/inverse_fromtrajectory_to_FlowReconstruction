"""Physics-balanced training; every printed training record includes unseen CFD error."""
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from .v2_profiles import sample
from .v2_model import StreamDeepONet, physics, outlet


def monitoring_errors(model,data,tensor):
    errors=[]
    model.eval()
    with torch.no_grad():
        for i in range(len(data['profiles'])):
            pred=model(tensor(data['profiles'][i:i+1]),tensor(data['reynolds'][i:i+1]),tensor(data['xy'][None])).cpu().numpy()[0]
            target=data['fields'][i];difference=pred-target
            errors.append({'case':i,'Re':float(data['reynolds'][i]),
                'u_rel_l2':float(np.linalg.norm(difference[:,0])/np.linalg.norm(target[:,0])),
                'v_rel_l2':float(np.linalg.norm(difference[:,1])/max(np.linalg.norm(target[:,1]),1e-12)),
                'v_rmse':float(np.sqrt(np.mean(difference[:,1]**2))),
                'p_rel_l2':float(np.linalg.norm(difference[:,2])/np.linalg.norm(target[:,2])),
                'velocity_rel_l2':float(np.linalg.norm(difference[:,:2])/np.linalg.norm(target[:,:2]))})
    model.train()
    return {'role':'validation_only_never_backpropagated',
            'velocity_rel_l2_max':max(c['velocity_rel_l2'] for c in errors),
            'p_rel_l2_max':max(c['p_rel_l2'] for c in errors),
            'velocity_rel_l2_mean':float(np.mean([c['velocity_rel_l2'] for c in errors])),
            'p_rel_l2_mean':float(np.mean([c['p_rel_l2'] for c in errors])),
            'cases':errors}


def monitoring_physics(model,data,tensor):
    rng=np.random.default_rng(81372)
    records=[]
    for i in range(len(data['profiles'])):
        p,r=tensor(data['profiles'][i:i+1]),tensor(data['reynolds'][i:i+1])
        xy=tensor(rng.uniform([.005,.005],[4.995,.995],(1,256,2)))
        mx,my,mass=physics(model,p,r,xy)
        on,ot=outlet(model,p,r,tensor(np.linspace(.001,.999,65)[None]))
        records.append({'case':i,'momentum_rms':float((mx.square().mean()+my.square().mean()).sqrt().detach()),
                        'divergence_rms':float(mass.square().mean().sqrt().detach()),
                        'outlet_rms':float((on.square().mean()+ot.square().mean()).sqrt().detach())})
    return {'momentum_rms_max':max(x['momentum_rms'] for x in records),'divergence_rms_max':max(x['divergence_rms'] for x in records),
            'outlet_rms_max':max(x['outlet_rms'] for x in records),'cases':records}


def train(output,steps=20000,log_every=250,batch=8,points=96,width=64,latent=32,depth=3,resume=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    seed=5197;torch.manual_seed(seed);rng=np.random.default_rng(seed);torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device=device)
    with np.load(output/'train.npz') as d: data={k:d[k].copy() for k in d.files}
    with np.load(output/'monitor.npz') as d: monitor={k:d[k].copy() for k in d.files}
    # Monitored coefficients never occur in the supervised training set.
    distance=np.abs(data['coefficients'][:,None,:]-monitor['coefficients'][None,:,:]).max(-1)
    if distance.min()<1e-8: raise ValueError('Monitor inlet leakage into training')
    td={k:tensor(data[k]) for k in ['profiles','reynolds','xy','fields']}
    model=StreamDeepONet(width,latent,depth).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    first=0;data_weight=1.
    if resume:
        state=torch.load(resume,map_location=device,weights_only=False)
        if state['config']!=model.config:raise ValueError('Resume architecture mismatch')
        model.load_state_dict(state['state_dict']);optimizer.load_state_dict(state['optimizer'])
        first=state['step'];data_weight=state['data_weight']
        if steps<=first:raise ValueError('Target steps must exceed resumed step')
        rng.bit_generator.state=state['numpy_rng_state'];torch.set_rng_state(state['torch_rng_state'].cpu())
        if device=='cuda' and state.get('cuda_rng_states') is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng_states'])
    params=list(model.parameters());start=time.time();history=[]
    log=open(output/'learning_curve.jsonl','a',buffering=1)
    start_record={'step':first,'seconds':0.,'validation':monitoring_errors(model,monitor,tensor),'device':device,'phase':'start'}
    print(json.dumps(start_record),flush=True);log.write(json.dumps(start_record)+'\n')
    grad_balance={};passed_checks=0
    for step in range(first+1,steps+1):
        model.train();optimizer.zero_grad(set_to_none=True)
        lr=max(1e-5,1e-3*.9**(step/1000))
        for group in optimizer.param_groups:group['lr']=lr
        ci=torch.randint(len(td['profiles']),(batch,),device=device)
        fresh,re,coeff=sample(rng,batch//2)
        pp=torch.cat([td['profiles'][ci[:batch-batch//2]],tensor(fresh)],0)
        rr=torch.cat([td['reynolds'][ci[:batch-batch//2]],tensor(re)],0)
        xy=torch.rand(batch,points,2,device=device)
        choose=torch.rand(batch,points,device=device)<.5
        xy[...,0]=5*torch.where(choose,xy[...,0]**2,xy[...,0])
        mx,my,mass=physics(model,pp,rr,xy)
        momentum=(mx.square()+my.square()).mean()
        on,ot=outlet(model,pp,rr,torch.rand(batch,32,device=device))
        outlet_loss=(on.square()+ot.square()).mean()
        physical=momentum+outlet_loss
        pi=torch.randint(len(td['xy']),(192,),device=device)
        prediction=model(td['profiles'][ci],td['reynolds'][ci],td['xy'][pi][None].expand(batch,-1,-1))
        target=td['fields'][ci[:,None],pi[None,:]]
        scale=torch.ones(batch,1,3,device=device)
        scale[...,1]=10 # lateral velocity is small but physically important
        scale[...,2]=td['reynolds'][ci,None]/60
        data_loss=((prediction-target)*scale).square().mean()
        if step==first+1 or step%250==0:
            def norm(loss):
                gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
                return sum(g.square().sum() for g in gs if g is not None).sqrt().detach()
            gp,gd=norm(physical),norm(data_loss)
            target_weight=float((gp/(gd+1e-12)).clamp(1e-3,10))
            data_weight=.8*data_weight+.2*target_weight
            grad_balance={'physical_gradient_l2':float(gp),'raw_data_gradient_l2':float(gd),'weighted_data_gradient_l2':float(gd)*data_weight}
        loss=physical+data_weight*data_loss
        if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss at {step}')
        loss.backward()
        grad_norm=float(torch.nn.utils.clip_grad_norm_(params,10.))
        optimizer.step()
        if step%log_every==0 or step==steps:
            record={'step':step,'seconds':time.time()-start,'loss':float(loss.detach()),'momentum_mse':float(momentum.detach()),
                    'outlet_mse':float(outlet_loss.detach()),'data_mse':float(data_loss.detach()),'data_weight':data_weight,
                    'divergence_rms':float(mass.square().mean().sqrt().detach()),'gradient_norm_before_clip':grad_norm,
                    'lr':lr,'gradient_balance':grad_balance,'validation':monitoring_errors(model,monitor,tensor)}
            print(json.dumps(record),flush=True);log.write(json.dumps(record)+'\n');history.append(record)
            stop=False
            if step%1000==0 or step==steps:
                diagnostic=monitoring_physics(model,monitor,tensor)
                record['validation_physics']=diagnostic
                passed=(record['validation']['velocity_rel_l2_max']<.05 and record['validation']['p_rel_l2_max']<.1
                        and diagnostic['momentum_rms_max']<.05 and diagnostic['outlet_rms_max']<.1)
                passed_checks=passed_checks+1 if passed else 0
                stop=step>=4000 and passed_checks>=2
                # Repeat metrics in this diagnostic line: every training print includes CFD errors.
                diagnostic_record={'step':step,'phase':'physics_diagnostic','validation':record['validation'],'validation_physics':diagnostic,'early_stop':stop}
                print(json.dumps(diagnostic_record),flush=True);log.write(json.dumps(diagnostic_record)+'\n')
            temporary=output/'v2_model.tmp'
            torch.save({'state_dict':model.state_dict(),'config':model.config,'optimizer':optimizer.state_dict(),'step':step,
                'model_type':'stream_deeponet_v2','data_weight':data_weight,'numpy_rng_state':rng.bit_generator.state,
                'torch_rng_state':torch.get_rng_state(),'cuda_rng_states':torch.cuda.get_rng_state_all() if device=='cuda' else None,
                'seed':seed,'torch_version':str(torch.__version__),'validation':record['validation']},temporary)
            temporary.replace(output/'v2_model.pt')
            if stop:break
    log.close()
    (output/'v2_training_summary.json').write_text(json.dumps({'device':device,'gpu':torch.cuda.get_device_name() if device=='cuda' else None,
        'planned_steps':steps,'steps':step,'seconds':time.time()-start,'final':history[-1] if history else start_record},indent=2))
    return model
