"""Physics-only training: inputs and NS/traction residuals, no CFD labels.

Reference metrics may be read by a separate monitor, never by loss/batch/scale.
"""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import argparse
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from .model import Config,initialize,make_functions
from .sampling import make_pool,diagnostic_profiles
from .checkpoint import save,load


def make_step(functions):
    def loss(params,batch):
        terms=functions['raw_terms'](params,batch)
        return terms[0]+10*terms[1],terms
    @jax.jit
    def step(params,m,v,count,batch):
        (loss_value,terms),grad=jax.value_and_grad(loss,has_aux=True)(params,batch)
        norm=jnp.sqrt(sum(jnp.sum(g*g) for g in jax.tree_util.tree_leaves(grad)))
        grad=jax.tree_util.tree_map(lambda g:g*jnp.minimum(1.,100/(norm+1e-12)),grad)
        m=jax.tree_util.tree_map(lambda a,g:.9*a+.1*g,m,grad)
        v=jax.tree_util.tree_map(lambda a,g:.999*a+.001*g*g,v,grad)
        lr=jnp.maximum(2e-5,1e-3*.95**(count/2000))
        params=jax.tree_util.tree_map(lambda p,a,b:p-lr*(a/(1-.9**count))/(jnp.sqrt(b/(1-.999**count))+1e-8),params,m,v)
        return params,m,v,loss_value,terms,norm
    return step


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
    config=Config(width=args.width,latent=args.latent,depth=args.depth)
    rng=np.random.default_rng(91838);params=initialize(config);first=0
    m,v=jax.tree_util.tree_map(jnp.zeros_like,params),jax.tree_util.tree_map(jnp.zeros_like,params)
    if args.resume:
        params,old_config,metadata,state=load(args.resume,optimizer=True)
        if old_config!=config:raise ValueError('Resume configuration differs from CLI configuration')
        if metadata['pool_seed']!=args.pool_seed or metadata['pool_count']!=args.pool_count:raise ValueError('Resume pool changed')
        first=metadata['step'];rng.bit_generator.state=metadata['rng_state'];m,v=state
    if args.steps<=first:raise ValueError('Target steps must exceed current checkpoint step')
    pool,pool_metadata=make_pool(args.pool_count,args.pool_seed,encoder_points=args.encoder_points)
    (output/'input_pool_metadata.json').write_text(json.dumps(pool_metadata,indent=2))
    pool_digest=hashlib.sha256(b''.join(v.tobytes() for v in pool.values())).hexdigest()
    if args.resume and metadata['pool_sha256']!=pool_digest:raise ValueError('Resume input pool or its representation changed')
    functions=make_functions(config);update=make_step(functions)
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
    points=vrng.uniform(size=(len(re),128,2)).astype(np.float32)
    points[...,0]=config.length*points[...,0]**2
    yy=np.broadcast_to(np.linspace(.001,.999,96,dtype=np.float32),(len(re),96)).copy()
    start=time.perf_counter()
    def monitor(step):
        residual=np.asarray(functions['residual'](params,held,re,points))
        boundary=np.asarray(functions['outlet'](params,held,re,yy))
        records=[dict(name=name,reynolds=float(r),momentum_rms=float(np.sqrt(np.mean(np.sum(rr[:,:2]**2,axis=-1)))),
                      divergence_rms=float(np.sqrt(np.mean(rr[:,2]**2))),
                      outlet_traction_rms=float(np.sqrt(np.mean(np.sum(oo**2,axis=-1)))))
                 for name,r,rr,oo in zip(names,re,residual,boundary)]
        return dict(step=step,seconds=time.perf_counter()-start,role='repeated_physics_validation_no_CFD_labels',cases=records)
    with (output/'training.jsonl').open('a',buffering=1) as log:
        initial=monitor(first);print(json.dumps(initial),flush=True);log.write(json.dumps(initial)+'\n')
        for count in range(first+1,args.steps+1):
            b=batch(rng,pool,args.batch,args.points)
            params,m,v,value,terms,gradient_norm=update(params,m,v,jnp.array(count),b)
            if count%args.log_every==0 or count==args.steps or count==first+1:
                record=monitor(count);record.update(loss=float(value),momentum_mse=float(terms[0]),
                    outlet_mse=float(terms[1]),divergence_mse=float(terms[2]),gradient_l2=float(gradient_norm))
                if not np.isfinite(record['loss']):raise RuntimeError('Non-finite objective')
                if reference_cases and (count%args.reference_every==0 or count==args.steps):
                    from .validation import compare
                    record['reference_validation']=compare(params,functions,reference_cases)
                print(json.dumps(record),flush=True);log.write(json.dumps(record)+'\n')
                save(output/'checkpoint.npz',params,config,count,rng.bit_generator.state,(m,v),
                    dict(pool_seed=args.pool_seed,pool_count=args.pool_count,pool_sha256=pool_digest,
                         encoder_points=args.encoder_points,training_mode='physics_only',CFD_response_labels_used=0,
                         validation_status='not_yet_independently_accepted',diagnostics=record))
    (output/'summary.json').write_text(json.dumps(dict(step=count,seconds=time.perf_counter()-start,
        devices=[str(d) for d in jax.devices()],config=config.json(),last=record,training_mode='physics_only',
        CFD_response_labels_used=0,validated_for_deployment=False),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=50000)
    p.add_argument('--batch',type=int,default=8);p.add_argument('--points',type=int,default=64)
    p.add_argument('--pool-count',type=int,default=10000);p.add_argument('--pool-seed',type=int,default=9703107)
    p.add_argument('--encoder-points',type=int,default=257);p.add_argument('--width',type=int,default=96)
    p.add_argument('--latent',type=int,default=48);p.add_argument('--depth',type=int,default=3)
    p.add_argument('--log-every',type=int,default=250);p.add_argument('--resume');p.add_argument('--require-gpu',action='store_true')
    p.add_argument('--reference-monitor');p.add_argument('--reference-every',type=int,default=1000)
    a=p.parse_args()
    if min(a.steps,a.batch,a.points,a.pool_count,a.log_every,a.reference_every,a.width,a.latent,a.depth)<1:p.error('Counts and network sizes must be positive')
    train(a)


if __name__=='__main__':main()
