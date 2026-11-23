"""Evaluate old SOAP fits on the same enlarged trajectory sets."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['JAX_ENABLE_X64']='true'
sys.dont_write_bytecode=True
from run import ARCHIVE, OUT, ROOT

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--flow',required=True,type=Path)
parser.add_argument('--cpu',type=int)
args = parser.parse_args()
if args.cpu is not None:
    os.sched_setaffinity(0,{args.cpu})
sys.path[:0] = [str(args.flow.resolve()),str(ROOT/'experiments/soap_particle_count')]

import jax.numpy as jnp
import numpy as np
from cached_operator import CachedObjective
from experiments.particle_count.fit_campaign import CampaignFitter, CONFIG
from experiments.particle_count import fit_campaign
from flow_reconstruction.particles import integral_observations
from flow_operator.model import context
from launches import read_pool

fit_campaign.CP = args.flow/'reports/experiments/soap_cosine_200k/final_checkpoint.npz'
fitter = CampaignFitter()
fast = CachedObjective(fitter,CONFIG)
model = fitter.m
output = []
for name,before,after in [('waves3_r30',4,6),('skew_b3',13,17),('lobes4_r30',20,24),('flat30',20,24)]:
    tracks = read_pool(ARCHIVE/'observations'/name)[:after]
    data = fast.data(tracks,integral_observations)
    rows = []
    for n in [before,after]:
        p = ARCHIVE/f'soap_314k/fits/{name}/N{n}/report.json'
        report = json.loads(p.read_text())
        sol = report['candidates'][report['selected_start']]
        theta = jnp.asarray(sol['theta'])
        objective,_ = fast.loss(theta,data,float(after))
        z,speed,re = model.decode(theta)
        profile = model.builder.arrays(z)
        branch = context(model.params,profile,re,model.operator.config)
        r = fast.positions(profile,speed,re,data,branch)-data['observed']
        rmse = float(jnp.sqrt(jnp.mean(r**2)))
        data_cost = 2*rmse**2/CONFIG.position_scale**2
        a,b = profile['coeff'][:2]
        roughness = float(CONFIG.smoothness*jnp.sum((12*a*a+12*a*b+4*b*b)/model.builder.h**3))
        rows.append(dict(fitted_N=n,evaluated_N=after,objective=float(objective),coordinate_rmse=rmse,
                         data_cost=data_cost,curvature_cost=roughness,other_penalties=float(objective)-data_cost-roughness,
                         inlet_l2_percent=100*report['inlet_relative_l2'],converged=sol['optimizer_success'],
                         iterations=sol['iterations'],stop_reason=sol['reason']))
    output.append(dict(profile=name,before=rows[0],after=rows[1]))
OUT.mkdir(exist_ok=True)
(OUT/'old_regressions.json').write_text(json.dumps(output,indent=2)+'\n')
for case in output:
    before,after=case['before'],case['after']
    print(case['profile'], 'same-data objective',before['objective'],after['objective'],
          'inlet error %',before['inlet_l2_percent'],after['inlet_l2_percent'])
