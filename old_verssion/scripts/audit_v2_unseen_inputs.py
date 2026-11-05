"""Replay host-side sampling to verify held-out profiles were never optimized.

For adaptive choice, the probabilities affect selected TRAIN ids but not the
number of uniform RNG draws; a uniform p consumes the identical RNG sequence.
No held-out response fields are loaded by this audit.
"""
import argparse,json,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from channel_operator.v2_profiles import sample


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--blind',required=True);p.add_argument('--output',required=True)
    p.add_argument('--steps',required=True,type=int);p.add_argument('--batch',type=int,default=16);p.add_argument('--points',type=int,default=96)
    p.add_argument('--resume-state');p.add_argument('--end-state')
    p.add_argument('--adaptive-physics',action='store_true');a=p.parse_args()
    data=Path(a.data)
    with np.load(data/'train.npz') as d:training=d['profiles'].astype(np.float32);nxy=len(d['xy'])
    with np.load(data/'monitor.npz') as d:monitor=d['profiles'].astype(np.float32)
    with np.load(a.blind) as d:blind=d['profiles'].astype(np.float32)
    held=np.concatenate([monitor,blind]);counts=np.zeros(len(held),dtype=int)
    for profile in training:counts+=np.all(profile[None]==held,axis=1)
    rng=np.random.default_rng(5197);first=0
    if a.resume_state:
        with np.load(a.resume_state,allow_pickle=False) as d:
            first=int(d['step']);rng.bit_generator.state=json.loads(str(d['rng_json']))
    total=0;n=a.batch-a.batch//2
    for step in range(first+1,a.steps+1):
        rng.integers(len(training),size=a.batch);rng.integers(nxy,size=192)
        fresh,_,_=sample(rng,a.batch//2);fresh=fresh.astype(np.float32)
        counts+=np.all(fresh[:,None,:]==held[None,:,:],axis=2).sum(0);total+=len(fresh)
        if a.adaptive_physics:rng.choice(len(training),size=n,p=np.full(len(training),1/len(training)))
        count=256 if a.adaptive_physics else a.points
        rng.uniform(size=(a.batch,count,2));rng.uniform(size=(a.batch,count))
        if a.adaptive_physics:
            for _ in range(a.batch):rng.choice(count,size=a.points,p=np.full(count,1/count))
        rng.uniform(size=(a.batch,32))
    rng_matches=None
    if a.end_state:
        with np.load(a.end_state,allow_pickle=False) as d:expected=json.loads(str(d['rng_json']))
        rng_matches=rng.bit_generator.state==expected
    result={'first_step':first,'last_step':a.steps,'fresh_physics_inlets_replayed':total,'training_profiles_checked':len(training),
        'monitor_profile_collisions':counts[:len(monitor)].tolist(),'blind_profile_collisions':counts[len(monitor):].tolist(),
        'all_heldout_profiles_unseen_by_optimizer':bool(not counts.any()),'response_fields_loaded':False,'adaptive_physics':a.adaptive_physics,
        'rng_replay_matches_saved_checkpoint':rng_matches,
        'training_file_sha256':hashlib.sha256((data/'train.npz').read_bytes()).hexdigest(),
        'blind_file_sha256':hashlib.sha256(Path(a.blind).read_bytes()).hexdigest()}
    Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
