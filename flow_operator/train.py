"""Physics-only training: inputs and NS/traction residuals, no CFD labels.

Reference metrics may be read by a separate monitor, never by loss/batch/scale.
"""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from .model import Config,initialize,make_functions
from .sampling import make_pool,diagnostic_profiles
from .checkpoint import save,load


def make_step(functions,normalization='none',axis_name=None):
    def loss(params,batch):
        if normalization in ['input_curvature','input_curvature_sqrt']:
            # Exact integral of (f'')^2 for each supplied cubic. This uses ONLY
            # input derivatives and Re, never response fields or CFD statistics.
            profile=batch['profiles'];a,b=profile['coeff'][:,0],profile['coeff'][:,1]
            h=jnp.diff(profile['knots'],axis=1)
            curvature=jnp.sqrt(jnp.maximum(jnp.sum((12*a*a+12*a*b+4*b*b)/h**3,axis=1),0))
            scale=1+curvature/batch['reynolds']
            r=functions['residual'](params,profile,batch['reynolds'],batch['xy'])
            o=functions['outlet'](params,profile,batch['reynolds'],batch['outlet_y'])
            exponent=2 if normalization=='input_curvature' else 1
            terms=jnp.array([jnp.mean(jnp.sum(r[...,:2]**2,axis=-1)/scale[:,None]**exponent),
                             jnp.mean(jnp.sum(o**2,axis=-1)),jnp.mean(r[...,2]**2)])
        else:
            terms=functions['raw_terms'](params,batch)
        return terms[0]+10*terms[1],terms
    def step(params,m,v,count,batch):
        (loss_value,terms),grad=jax.value_and_grad(loss,has_aux=True)(params,batch)
        if axis_name is not None:
            grad=jax.lax.pmean(grad,axis_name)
            loss_value=jax.lax.pmean(loss_value,axis_name)
            terms=jax.lax.pmean(terms,axis_name)
        norm=jnp.sqrt(sum(jnp.sum(g*g) for g in jax.tree_util.tree_leaves(grad)))
        grad=jax.tree_util.tree_map(lambda g:g*jnp.minimum(1.,100/(norm+1e-12)),grad)
        m=jax.tree_util.tree_map(lambda a,g:.9*a+.1*g,m,grad)
        v=jax.tree_util.tree_map(lambda a,g:.999*a+.001*g*g,v,grad)
        lr=jnp.maximum(2e-5,1e-3*.95**(count/2000))
        params=jax.tree_util.tree_map(lambda p,a,b:p-lr*(a/(1-.9**count))/(jnp.sqrt(b/(1-.999**count))+1e-8),params,m,v)
        return params,m,v,loss_value,terms,norm
    return jax.jit(step) if axis_name is None else jax.pmap(step,axis_name=axis_name)


def replicate(tree,devices):
    return jax.tree_util.tree_map(lambda x:jnp.broadcast_to(x,(devices,)+x.shape),tree)


def shard_batch(data,devices):
    return jax.tree_util.tree_map(lambda x:x.reshape(devices,len(x)//devices,*x.shape[1:]),data)


def batch(rng,pool,batch_size,points,length=5.):
    indices=rng.integers(len(pool['knots']),size=batch_size)
    profiles={k:v[indices] for k,v in pool.items()}
    re=np.exp(rng.uniform(np.log(10),np.log(100),batch_size)).astype(np.float32)
    # Half of conditions explicitly cover both ends of the Re range.
    edge=rng.random(batch_size)<.2;re[edge]=rng.choice([10.,100.],edge.sum())
    xy=rng.uniform(size=(batch_size,points,2)).astype(np.float32)
    # Cover the full domain and the short development lengths of rough inlets.
    choose=rng.random(xy.shape[:-1])<.65
    xy[...,0]=length*np.where(choose,xy[...,0]**3,xy[...,0])
    return dict(profiles=profiles,reynolds=re,xy=xy,
                outlet_y=rng.uniform(size=(batch_size,48)).astype(np.float32))


def train(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    if args.require_gpu and not any(d.platform=='gpu' for d in jax.devices()):raise RuntimeError('GPU required')
    if args.devices>len(jax.local_devices()) or args.batch%args.devices:
        raise ValueError('Requested devices unavailable or batch not divisible by devices')
    best_score=float('inf')
    config=Config(width=args.width,latent=args.latent,depth=args.depth,lift=args.lift,functional_moments=args.functional_moments,
                  encoder_kind=args.encoder_kind,sensor_count=args.sensor_count,separate_pressure_trunk=args.separate_pressure_trunk)
    rng=np.random.default_rng(91838);params=initialize(config);first=0
    m,v=jax.tree_util.tree_map(jnp.zeros_like,params),jax.tree_util.tree_map(jnp.zeros_like,params)
    if args.resume:
        params,old_config,metadata,state=load(args.resume,optimizer=True)
        if old_config!=config:raise ValueError('Resume configuration differs from CLI configuration')
        if metadata['pool_seed']!=args.pool_seed or metadata['pool_count']!=args.pool_count:raise ValueError('Resume pool changed')
        first=metadata['step'];rng.bit_generator.state=metadata['rng_state'];m,v=state
        previous_best=Path(args.resume).parent/'best_checkpoint.npz'
        if previous_best.exists():
            if previous_best.resolve()!=(output/'best_checkpoint.npz').resolve():shutil.copy2(previous_best,output/'best_checkpoint.npz')
            _,_,best_metadata,_=load(previous_best)
            same_monitor=(best_metadata.get('physics_monitor_points',128)==args.monitor_points and
                          best_metadata.get('reference_used_for_selection',False)==bool(args.reference_monitor))
            if same_monitor:best_score=best_metadata.get('selection_score',float('inf'))
    if args.steps<=first:raise ValueError('Target steps must exceed current checkpoint step')
    pool,pool_metadata=make_pool(args.pool_count,args.pool_seed,encoder_points=args.encoder_points)
    (output/'input_pool_metadata.json').write_text(json.dumps(pool_metadata,indent=2))
    pool_digest=hashlib.sha256(b''.join(v.tobytes() for v in pool.values())).hexdigest()
    if args.resume and metadata['pool_sha256']!=pool_digest:raise ValueError('Resume input pool or its representation changed')
    functions=make_functions(config);parallel=args.devices>1
    update=make_step(functions,args.physics_normalization,'training_devices' if parallel else None)
    if parallel:params,m,v=replicate(params,args.devices),replicate(m,args.devices),replicate(v,args.devices)
    first_replica=lambda tree:jax.tree_util.tree_map(lambda x:x[0],tree) if parallel else tree
    reference_cases=None
    if args.reference_monitor:
        from .validation import read_cases
        reference_cases=read_cases(args.reference_monitor,args.encoder_points)
    held,re,names=diagnostic_profiles(encoder_points=args.encoder_points)
    for i,name in enumerate(names):
        if name!='poiseuille' and np.any(np.all(pool['coeff']==held['coeff'][i],axis=(1,2))):
            raise ValueError(f'Exact diagnostic profile leaked into the input training bank: {name}')
    vrng=np.random.default_rng(918390)
    # Fixed and independent of the optimizer RNG, with entrance coverage.
    points=vrng.uniform(size=(len(re),args.monitor_points,2)).astype(np.float32)
    points[...,0]=config.length*points[...,0]**2
    yy=np.broadcast_to(np.linspace(.001,.999,96,dtype=np.float32),(len(re),96)).copy()
    start=time.perf_counter()
    def monitor(step):
        current=first_replica(params)
        residual=np.asarray(functions['residual'](current,held,re,points))
        boundary=np.asarray(functions['outlet'](current,held,re,yy))
        records=[dict(name=name,reynolds=float(r),momentum_rms=float(np.sqrt(np.mean(np.sum(rr[:,:2]**2,axis=-1)))),
                      divergence_rms=float(np.sqrt(np.mean(rr[:,2]**2))),
                      outlet_traction_rms=float(np.sqrt(np.mean(np.sum(oo**2,axis=-1)))))
                 for name,r,rr,oo in zip(names,re,residual,boundary)]
        return dict(step=step,seconds=time.perf_counter()-start,role='repeated_physics_validation_no_CFD_labels',
                    physics_monitor_points=args.monitor_points,cases=records)
    with (output/'training.jsonl').open('a',buffering=1) as log:
        initial=monitor(first);print(json.dumps(initial),flush=True);log.write(json.dumps(initial)+'\n')
        for count in range(first+1,args.steps+1):
            b=batch(rng,pool,args.batch,args.points)
            if parallel:
                b=shard_batch(b,args.devices);counter=jnp.full((args.devices,),count)
            else:counter=jnp.array(count)
            params,m,v,value,terms,gradient_norm=update(params,m,v,counter,b)
            if count%args.log_every==0 or count==args.steps or count==first+1:
                objective=first_replica(terms)
                record=monitor(count);record.update(loss=float(first_replica(value)),momentum_mse=float(objective[0]),
                    outlet_mse=float(objective[1]),divergence_mse=float(objective[2]),gradient_l2=float(first_replica(gradient_norm)),
                    physics_normalization=args.physics_normalization,
                    momentum_mse_is_normalized=args.physics_normalization!='none')
                if not np.isfinite(record['loss']):raise RuntimeError('Non-finite objective')
                if reference_cases and (count%args.reference_every==0 or count==args.steps):
                    from .validation import compare
                    record['reference_validation']=compare(first_replica(params),functions,reference_cases)
                print(json.dumps(record),flush=True);log.write(json.dumps(record)+'\n')
                physical_score=max(max(c['momentum_rms']/.05,c['outlet_traction_rms']/.01,c['divergence_rms']/1e-4) for c in record['cases'])
                score=physical_score
                used_reference='reference_validation' in record
                if used_reference:
                    score=max(score,record['reference_validation']['max_velocity_relative_l2']/.05,
                              record['reference_validation']['max_pressure_relative_l2']/.1)
                checkpoint_metadata=dict(pool_seed=args.pool_seed,pool_count=args.pool_count,pool_sha256=pool_digest,
                         encoder_points=args.encoder_points,training_mode='physics_only',CFD_response_labels_used=0,
                         physics_normalization=args.physics_normalization,
                         training_devices=args.devices,global_batch=args.batch,physics_points_per_condition=args.points,
                         physics_monitor_points=args.monitor_points,
                         validation_status='not_yet_independently_accepted',diagnostics=record,
                         selection_score=score,reference_used_for_selection=used_reference)
                save(output/'checkpoint.npz',first_replica(params),config,count,rng.bit_generator.state,(first_replica(m),first_replica(v)),checkpoint_metadata)
                # Reference-enabled runs compare checkpoints only at reference
                # validation times, so the score has a consistent definition.
                if (reference_cases is None or used_reference) and score<best_score:
                    best_score=score;shutil.copy2(output/'checkpoint.npz',output/'best_checkpoint.npz')
    (output/'summary.json').write_text(json.dumps(dict(step=count,seconds=time.perf_counter()-start,
        devices=[str(d) for d in jax.devices()],config=config.json(),last=record,training_mode='physics_only',
        CFD_response_labels_used=0,validated_for_deployment=False),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=50000)
    p.add_argument('--batch',type=int,default=64);p.add_argument('--points',type=int,default=48)
    p.add_argument('--pool-count',type=int,default=10000);p.add_argument('--pool-seed',type=int,default=9703107)
    p.add_argument('--encoder-points',type=int,default=257);p.add_argument('--width',type=int,default=128)
    p.add_argument('--latent',type=int,default=64);p.add_argument('--depth',type=int,default=4)
    p.add_argument('--log-every',type=int,default=250);p.add_argument('--resume');p.add_argument('--require-gpu',action='store_true')
    p.add_argument('--reference-monitor');p.add_argument('--reference-every',type=int,default=1000)
    p.add_argument('--lift',choices=['gaussian','oseen'],default='oseen')
    p.add_argument('--physics-normalization',choices=['none','input_curvature','input_curvature_sqrt'],default='input_curvature_sqrt')
    p.add_argument('--functional-moments',action='store_true')
    p.add_argument('--encoder-kind',choices=['pooled','sensor'],default='sensor')
    p.add_argument('--sensor-count',type=int,default=257)
    p.add_argument('--separate-pressure-trunk',action='store_true')
    p.add_argument('--devices',type=int,default=1)
    p.add_argument('--monitor-points',type=int,default=1024)
    a=p.parse_args()
    if min(a.steps,a.batch,a.points,a.pool_count,a.log_every,a.reference_every,a.width,a.latent,a.depth,a.devices,a.monitor_points)<1:p.error('Counts and network sizes must be positive')
    train(a)


if __name__=='__main__':main()
