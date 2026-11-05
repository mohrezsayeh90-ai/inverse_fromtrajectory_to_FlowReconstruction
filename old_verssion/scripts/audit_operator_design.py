"""Audit current operator restrictions against pinned upstream notebook sources.

Read-only with respect to models/training data. Writes diagnostic evidence only.
No optimizer step, CFD solve, or training job is started.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.interpolate import CubicSpline
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from channel_operator.profiles import Y
from channel_operator.v2_model import load
from channel_operator.v2_profiles import inlet_values

UPSTREAM_COMMIT = '77e7c747d653b34e79e43df7d88bad87de5c27d8'
NOTEBOOKS = [
    'Antiderivative/PI_DeepONet_FF_antideriv.ipynb',
    'Antiderivative/PI_DeepONet_antideriv.ipynb',
    'Burger/PI_DeepONet_Burger.ipynb',
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def imports(path):
    return [n.module for n in ast.walk(ast.parse(path.read_text()))
            if isinstance(n, ast.ImportFrom)]


def reference_snapshots(upstream, output):
    snapshot = output/'upstream_excerpts.json'
    if not upstream.exists():
        saved = json.loads(snapshot.read_text())
        assert saved['commit'] == UPSTREAM_COMMIT
        return saved
    commit = subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'], text=True).strip()
    assert commit == UPSTREAM_COMMIT, 'Upstream must match the reviewed commit'
    saved = dict(repository='https://github.com/PredictiveIntelligenceLab/Physics-informed-DeepONets',
                 commit=commit, note='Selected source cells only; no notebook outputs or trained weights.', notebooks={})
    for name in NOTEBOOKS:
        path = upstream/name
        nb = json.loads(path.read_text())
        cells = []
        for index, cell in enumerate(nb['cells']):
            source = ''.join(cell.get('source', []))
            if cell['cell_type'] == 'code' and any(k in source for k in [
                    'def MLP', 'class PI_DeepONet', 'def RBF', 'generate_one_ics_training_data',
                    'N_train =', 'N_train =1000', 'model.train(', 'key_test =', 'u0_train =']):
                cells.append(dict(cell_index_zero_based=index, source=source))
        saved['notebooks'][name] = dict(sha256=digest(path),
            url=saved['repository']+'/blob/'+commit+'/'+name, cells=cells)
    snapshot.write_text(json.dumps(saved, indent=2))
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, default=ROOT.parent/'reference-physics-informed-deeponets')
    parser.add_argument('--output', type=Path, default=ROOT/'reports/operator_design_audit')
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    snapshot = reference_snapshots(args.upstream, args.output)
    torch.set_num_threads(4)
    checkpoint = ROOT/'models/v2/stream_operator.pt'; sha = digest(checkpoint)
    model, state = load(checkpoint); model.double()
    y = np.linspace(0., 1., 1025)
    nine = lambda yy: 6*yy*(1-yy)*(1-.6*np.cos(18*np.pi*yy))/(1+7.2/(18*np.pi)**2)
    t = lambda a: torch.tensor(a, dtype=torch.float64)
    xy = t(np.column_stack([np.zeros_like(y),y])[None])
    predicted = model(t(nine(Y)[None]), t([55.]), xy)
    target = t(nine(y))
    loss = ((predicted[0,:,0]-target)**2+predicted[0,:,1]**2).mean()
    grads = torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)
    gradient_norm = float(torch.sqrt(sum((g*g).sum() for g in grads if g is not None)))
    observed = predicted.detach().numpy()[0,:,0]
    relative_error = float(np.sqrt(np.trapezoid((observed-nine(y))**2,y)/np.trapezoid(nine(y)**2,y)))
    assert gradient_norm == 0 and .38 < relative_error < .40
    controls = []
    for a in [[.25,-.1,.05,0.], [0.,0.,0.,.6]]:
        with torch.no_grad():
            field = model(t(inlet_values(a)[None]),t([55.]),xy).numpy()[0]
        error = float(np.max(abs(field[:,0]-inlet_values(a,y))))
        assert error < 1e-6
        controls.append(dict(coefficients=a, inlet_max_error=error))
    aliases = []
    dense = np.linspace(0.,1.,32769)
    base = lambda yy: 6*yy*(1-yy)
    for m in [33,257]:
        sensors = np.linspace(0.,1.,m)
        other = lambda yy: base(yy)*(1+.4*np.sin(np.pi*(m-1)*yy))
        same_samples_error = float(np.max(abs(base(sensors)-other(sensors))))
        hidden_difference = float(np.sqrt(np.trapezoid((base(dense)-other(dense))**2,dense)
                                          /np.trapezoid(base(dense)**2,dense)))
        assert same_samples_error < 1e-12 and hidden_difference > .28
        aliases.append(dict(sensors=m, formula='f0=6*y*(1-y); f1=f0*(1+0.4*sin(pi*(m-1)*y))',
            sensor_max_difference=same_samples_error, full_profile_relative_l2_difference=hidden_difference,
            mean_f0=float(np.trapezoid(base(dense),dense)),mean_f1=float(np.trapezoid(other(dense),dense)),
            min_f1=float(other(dense).min()),max_f1=float(other(dense).max())))
    split_counts = {}
    for name in ['train','monitor','blind']:
        with np.load(ROOT/f'data/v2/{name}.npz') as d:
            split_counts[name] = dict(conditions=len(d['profiles']), unique_profiles=len(np.unique(d['profiles'],axis=0)))
    trainer = ROOT/'run_v2_jax.py'
    trainer_imports = imports(trainer)
    assert 'channel_operator.v2_profiles' in trainer_imports
    assert 'channel_operator.broad_profiles' not in trainer_imports
    with np.load(ROOT/'data/broad_inlets/train.npz') as d:
        broad_shape = list(d['profiles'].shape)
    assert broad_shape == [10000,257] and model.branch.u.in_features == 34
    # A smooth bounded profile excluded by the newly generated positive envelope.
    yy = np.array([.01,.05,.1,.5])
    excluded = 630*yy**4*(1-yy)**4
    ratio = excluded/base(yy)
    assert np.any(ratio < 1/7)
    ff = '\n'.join(c['source'] for c in snapshot['notebooks'][NOTEBOOKS[0]]['cells'])
    basic_cells = snapshot['notebooks'][NOTEBOOKS[1]]['cells']
    basic_settings = next(c['source'] for c in basic_cells if 'N_train = 10000' in c['source'])
    assert 'y_train = np.zeros' in ff and 's_train = np.zeros' in ff
    assert '# y_train = np.zeros' in basic_settings and '# s_train = np.zeros' in basic_settings
    source_files = ['channel_operator/model.py','channel_operator/profiles.py','channel_operator/train.py',
        'channel_operator/v2_model.py','channel_operator/v2_jax.py','channel_operator/v2_profiles.py',
        'channel_operator/v2_predict.py','channel_operator/v2_train.py','channel_operator/v2_evaluate.py',
        'channel_operator/v2_data.py','channel_operator/broad_profiles.py','run_v2_jax.py','run_experiment.py']
    published = json.loads((ROOT/'models/v2/metadata.json').read_text())
    source_matches = {p:digest(ROOT/p)==expected for p,expected in published['sources'].items()}
    assert all(source_matches.values()), 'Current V2 source differs from saved checkpoint provenance'
    evidence = dict(upstream_commit=UPSTREAM_COMMIT, checkpoint_sha256=sha, checkpoint_step=state['step'],
        optimizer_steps_performed=0, cfd_solves_performed=0,
        source_sha256={p:digest(ROOT/p) for p in source_files},
        current_v2_sources_match_published_metadata=source_matches,
        nine_cycle_boundary=dict(relative_l2_error=relative_error, boundary_loss=float(loss.detach()),
            gradient_l2_wrt_all_trainable_parameters=gradient_norm,
            conclusion='The current inlet error cannot be changed by optimizing network weights.',
            cubic_spline_min_from_33_samples=float(CubicSpline(Y,nine(Y))(y).min()),
            cubic_spline_min_from_257_samples=float(CubicSpline(np.linspace(0,1,257),nine(np.linspace(0,1,257)))(y).min())),
        four_mode_boundary_controls=controls, finite_sensor_counterexamples=aliases,
        actual_v2_splits=split_counts, compiled_trainer_imports=trainer_imports,
        broad_bank_shape=broad_shape, current_branch_input_features=model.branch.u.in_features,
        broad_generator_envelope_counterexample=dict(formula='630*y^4*(1-y)^4', mean=1., max_speed=630/256,
            coordinate_samples=yy.tolist(),ratio_to_parabolic_base=ratio.tolist(),
            generator_global_lower_ratio_bound=1/7,
            conclusion='Even the broad bank has a restricted envelope; it is not all admissible profiles.'),
        upstream_notebook_caveat=dict(
            basic_default='One interior response label per input; boundary-only override is commented out.',
            fourier_default='Interior labels are computed by the shared generator but overwritten by s(0)=0 before training.',
            burgers='Training loss uses initial data, periodic boundary constraints and PDE residual; the provided train call is commented out.',
            source='Pinned notebook source, not inference from the filename.'))
    assert digest(checkpoint) == sha
    (args.output/'evidence.json').write_text(json.dumps(evidence, indent=2))
    print(json.dumps({k:v for k,v in evidence.items() if k not in ['source_sha256','compiled_trainer_imports']},indent=2))


if __name__=='__main__':
    main()
