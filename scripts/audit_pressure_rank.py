"""Distinguish pressure separation-rank capacity from optimizer performance.

The oracle projection uses monitoring responses ONLY for a mathematical span
bound, never for predictor training. The separate ridge predictor is train-only.
"""
import argparse,json,itertools
from pathlib import Path
import numpy as np


def features(a,re):
    x=np.column_stack([a/.8,2*np.log(re/10)/np.log(10)-1])
    columns=[np.ones(len(x))]
    for degree in range(1,4):
        for indices in itertools.combinations_with_replacement(range(5),degree):
            columns.append(np.prod(x[:,indices],axis=1))
    return np.column_stack(columns)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    data=Path(a.data)
    with np.load(data/'train.npz') as d:train={k:d[k].astype(float) for k in d.files}
    with np.load(data/'monitor.npz') as d:monitor={k:d[k].astype(float) for k in d.files}
    selected=json.loads((data/'packaging.json').read_text())['selected_train_point_indices']
    if not np.allclose(monitor['xy'][selected],train['xy']):raise ValueError('Coordinate mismatch')
    x=train['xy'][:,0]
    reference=monitor['fields'][:,selected,2]
    ft=features(train['coefficients'],train['reynolds']);fm=features(monitor['coefficients'],monitor['reynolds'])
    results={'note':'Rank oracle uses monitor responses to measure representability, not predictive accuracy. Ridge is fitted only to training labels. All comparisons use the same 256 packed anchor coordinates.','modes':{}}
    for mode in ['viscous','convective']:
        st=12/train['reynolds'][:,None];sm=12/monitor['reynolds'][:,None]
        raw=train['fields'][...,2]-st*(5-x)
        held=reference-sm*(5-x)
        if mode=='viscous':raw=raw/st;held=held/sm
        _,s,vh=np.linalg.svd(raw,full_matrices=False)
        basis=vh[:32]
        oracle=held@basis.T@basis
        ridge=np.linalg.solve(ft.T@ft+1e-6*np.eye(ft.shape[1]),ft.T@raw)
        prediction=fm@ridge
        if mode=='viscous':oracle=oracle*sm;prediction=prediction*sm
        oracle+=sm*(5-x);prediction+=sm*(5-x)
        oracle_errors=np.linalg.norm(oracle-reference,axis=1)/np.linalg.norm(reference,axis=1)
        predicted_errors=np.linalg.norm(prediction-reference,axis=1)/np.linalg.norm(reference,axis=1)
        results['modes'][mode]={'rank32_training_relative_frobenius_residual':float(np.sqrt(np.sum(s[32:]**2)/np.sum(s*s))),
            'rank32_oracle_monitor_pressure_error_max':float(oracle_errors.max()),'rank32_oracle_monitor_pressure_errors':oracle_errors.tolist(),
            'train_only_cubic_ridge_monitor_pressure_error_max':float(predicted_errors.max()),'train_only_cubic_ridge_monitor_pressure_errors':predicted_errors.tolist()}
    Path(a.output).write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))


if __name__=='__main__':main()
