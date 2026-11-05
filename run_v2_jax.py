"""Compiled training with monitored CFD labels excluded from every gradient."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import argparse,json,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from channel_operator.v2_jax import initialize,make_functions
from channel_operator.v2_profiles import sample


def train(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    devices=jax.devices()
    if args.require_gpu and not any(d.platform=='gpu' for d in devices):raise RuntimeError('GPU required')
    with np.load(Path(args.data)/'train.npz') as d:train={k:d[k].astype(np.float32) for k in d.files}
    with np.load(Path(args.data)/'monitor.npz') as d:monitor={k:d[k].astype(np.float32) for k in d.files}
    if np.min(np.abs(train['coefficients'][:,None]-monitor['coefficients'][None]).max(-1))<1e-8:raise ValueError('Monitor leakage')
    config=dict(width=args.width,latent=args.latent,depth=args.depth)
    if args.pressure_mode!='viscous':config['pressure_mode']=args.pressure_mode
    params=initialize(**config);m=jax.tree_util.tree_map(jnp.zeros_like,params);v=jax.tree_util.tree_map(jnp.zeros_like,params)
    rng=np.random.default_rng(5197);weight=1.;first=0
    if args.resume:
        with np.load(args.resume,allow_pickle=False) as s:
            params={k:jnp.asarray(s['param::'+k]) for k in params};m={k:jnp.asarray(s['m::'+k]) for k in params};v={k:jnp.asarray(s['v::'+k]) for k in params}
            first=int(s['step']);weight=float(s['data_weight']);rng.bit_generator.state=json.loads(str(s['rng_json']))
    fns=make_functions(args.latent,args.depth,args.pressure_mode)
    log=open(output/'learning_curve.jsonl','a',buffering=1);start=time.time();checks=0
    monitor_xy=np.broadcast_to(monitor['xy'],(len(monitor['profiles']),len(monitor['xy']),2)).copy()
    def validation():
        pred=np.asarray(fns['fields'](params,monitor['profiles'],monitor['reynolds'],monitor_xy))
        records=[]
        for i in range(len(pred)):
            truth=monitor['fields'][i];d=pred[i]-truth
            records.append({'case':i,'Re':float(monitor['reynolds'][i]),'u_rel_l2':float(np.linalg.norm(d[:,0])/np.linalg.norm(truth[:,0])),
                'v_rel_l2':float(np.linalg.norm(d[:,1])/max(np.linalg.norm(truth[:,1]),1e-12)),'v_rmse':float(np.sqrt(np.mean(d[:,1]**2))),
                'p_rel_l2':float(np.linalg.norm(d[:,2])/np.linalg.norm(truth[:,2])),
                'velocity_rel_l2':float(np.linalg.norm(d[:,:2])/np.linalg.norm(truth[:,:2]))})
        return {'role':'validation_only_never_backpropagated','cases':records,'velocity_rel_l2_max':max(c['velocity_rel_l2'] for c in records),
            'p_rel_l2_max':max(c['p_rel_l2'] for c in records),'velocity_rel_l2_mean':float(np.mean([c['velocity_rel_l2'] for c in records])),
            'p_rel_l2_mean':float(np.mean([c['p_rel_l2'] for c in records]))}
    def physical_validation():
        prng=np.random.default_rng(81372);xy=prng.uniform([.005,.005],[4.995,.995],(len(monitor['profiles']),256,2)).astype(np.float32)
        residual=np.asarray(fns['physics'](params,monitor['profiles'],monitor['reynolds'],xy))
        y=np.broadcast_to(np.linspace(.001,.999,65,dtype=np.float32),(len(monitor['profiles']),65)).copy()
        out=np.asarray(fns['outlet'](params,monitor['profiles'],monitor['reynolds'],y))
        mom=np.sqrt(np.mean(np.sum(residual[...,:2]**2,-1),-1));mass=np.sqrt(np.mean(residual[...,2]**2,-1));bc=np.sqrt(np.mean(np.sum(out*out,-1),-1))
        return {'momentum_rms_max':float(mom.max()),'divergence_rms_max':float(mass.max()),'outlet_rms_max':float(bc.max()),
            'cases':[{'case':i,'momentum_rms':float(mom[i]),'divergence_rms':float(mass[i]),'outlet_rms':float(bc[i])} for i in range(len(mom))]}
    def emit(record):
        print(json.dumps(record),flush=True);log.write(json.dumps(record)+'\n')
    emit({'step':first,'phase':'start','seconds':time.time()-start,'backend':'jax','devices':[str(x) for x in devices],'validation':validation()})
    def save(step,record):
        arrays={}
        for prefix,tree in [('param',params),('m',m),('v',v)]:
            arrays.update({prefix+'::'+k:np.asarray(value) for k,value in tree.items()})
        arrays.update(step=np.array(step),data_weight=np.array(weight),rng_json=np.array(json.dumps(rng.bit_generator.state)),config_json=np.array(json.dumps(config)))
        np.savez_compressed(output/'jax_checkpoint.npz',**arrays)
        (output/'latest_validation.json').write_text(json.dumps(record,indent=2))
    gradient_balance={};last=None
    for step in range(first+1,args.steps+1):
        ci=rng.integers(len(train['profiles']),size=args.batch);pi=rng.integers(len(train['xy']),size=192)
        fresh,re,_=sample(rng,args.batch//2)
        n=args.batch-args.batch//2
        pp=np.concatenate([train['profiles'][ci[:n]],fresh]).astype(np.float32)
        rr=np.concatenate([train['reynolds'][ci[:n]],re]).astype(np.float32)
        xy=rng.uniform(size=(args.batch,args.points,2)).astype(np.float32);choose=rng.uniform(size=xy.shape[:-1])<.5
        xy[...,0]=5*np.where(choose,xy[...,0]**2,xy[...,0])
        batch={'profiles':pp,'reynolds':rr,'xy':xy,'out_y':rng.uniform(size=(args.batch,32)).astype(np.float32),
            'data_profiles':train['profiles'][ci],'data_reynolds':train['reynolds'][ci],
            'data_xy':np.broadcast_to(train['xy'][pi],(args.batch,len(pi),2)).copy(),'target':train['fields'][ci[:,None],pi[None,:]]}
        if step==first+1 or step%250==0:
            gp,gd=fns['balance'](params,batch);gp,gd=float(gp),float(gd)
            weight=.8*weight+.2*np.clip(gp/(gd+1e-12),1e-3,10)
            gradient_balance={'physical_gradient_l2':gp,'raw_data_gradient_l2':gd,'weighted_data_gradient_l2':weight*gd}
        params,m,v,value,aux,gn=fns['step'](params,m,v,jnp.array(step),jnp.array(weight,dtype=jnp.float32),batch)
        if step%args.log_every==0 or step==args.steps:
            value=float(value);aux=np.asarray(aux)
            if not np.isfinite(value):raise RuntimeError('Nonfinite loss')
            record={'step':step,'seconds':time.time()-start,'loss':value,'momentum_mse':float(aux[0]),'outlet_mse':float(aux[1]),
                'data_mse':float(aux[2]),'divergence_rms':float(aux[3]),'data_weight':float(weight),'gradient_norm_before_clip':float(gn),
                'gradient_balance':gradient_balance,'validation':validation()}
            stop=False
            if step%1000==0 or step==args.steps:
                phy=physical_validation();record['validation_physics']=phy
                passed=(record['validation']['velocity_rel_l2_max']<.05 and record['validation']['p_rel_l2_max']<.1 and phy['momentum_rms_max']<.05 and phy['outlet_rms_max']<.1)
                checks=checks+1 if passed else 0;stop=step>=4000 and checks>=2
                record['early_stop']=stop
            emit(record);save(step,record);last=record
            if stop:break
    log.close()
    # Export to the numerically parity-tested PyTorch inference class; optimizer
    # remains in the JAX checkpoint, not in this portable inference-only file.
    import torch
    from channel_operator.v2_model import StreamDeepONet
    model=StreamDeepONet(**config);state=model.state_dict()
    state.update({k:torch.from_numpy(np.array(value)) for k,value in params.items()})
    model.load_state_dict(state)
    torch.save({'state_dict':model.state_dict(),'config':config,'step':step,'model_type':'stream_deeponet_v2_jax','data_weight':float(weight)},output/'v2_model.pt')
    summary={'backend':'jax','jax_version':jax.__version__,'devices':[str(x) for x in devices],'steps':step,'planned_steps':args.steps,'seconds':time.time()-start,'config':config,'final':last}
    (output/'v2_training_summary.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=50000);p.add_argument('--log-every',type=int,default=250)
    p.add_argument('--batch',type=int,default=16);p.add_argument('--points',type=int,default=96)
    p.add_argument('--width',type=int,default=64);p.add_argument('--latent',type=int,default=32);p.add_argument('--depth',type=int,default=3)
    p.add_argument('--require-gpu',action='store_true');p.add_argument('--resume')
    p.add_argument('--pressure-mode',choices=['viscous','convective'],default='viscous')
    args=p.parse_args()
    if args.batch<2 or min(args.steps,args.log_every,args.points)<1:p.error('Invalid counts')
    train(args)
