"""Recover the inlet directly from the early portions of near-inlet tracks."""
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import lsq_linear

from flow_reconstruction.particles import integral_observations

VALIDATION_EDGE = .04


@dataclass(frozen=True)
class LocalFit:
    knots: int
    edge: float
    quadratic: bool
    smoothness: float


def spline_basis(y, size):
    knots = np.r_[np.zeros(3),np.linspace(0,1,size-2),np.ones(3)]
    return BSpline.design_matrix(y,knots,3).toarray()*(6*y*(1-y))[:,None]


def system(tracks, basis_size, region, correction_scale, quadratic):
    designs, readings = [], []
    for track in tracks:
        integ = integral_observations([track],5,3)
        x,y = integ['xy'].T
        basis = spline_basis(y,basis_size)
        if quadratic:
            basis = np.column_stack([basis,basis*(x/correction_scale)[:,None]**2])
        # Keep only observations whose complete integration history is near the inlet.
        mask = (np.maximum.accumulate(track.position[:,0])[1:]<=region) & (track.time[1:]>=.055)
        designs.append((integ['weights']@basis)[mask])
        readings.append((integ['observed']-integ['known'])[mask,0])
    return np.concatenate(designs),np.concatenate(readings)


def estimate(training, validation, heights, resolutions=(11,21,29,37)):
    candidates = []
    for size in resolutions:
        diff = np.diff(np.eye(size),n=2,axis=0)
        for region in (.026,.045,.075):
            for quadratic in (False,True):
                matrix,observed = system(training,size,region,region,quadratic)
                if len(observed)<size:
                    continue
                check,target = system(validation,size,VALIDATION_EDGE,region,quadratic)
                if quadratic:
                    zer = np.zeros_like(diff)
                    nuisance = np.column_stack([np.zeros((size,size)),.25*np.eye(size)])
                    penalty = np.vstack([np.block([[diff,zer],[zer,diff]]),nuisance])
                    lower = np.r_[np.zeros(size),np.full(size,-4.)]
                    upper = np.r_[np.full(size,5.),np.full(size,4.)]
                else:
                    penalty = diff
                    lower,upper = 0.,5.
                for smoothness in (.00005,.0003,.0015,.006):
                    # Roughness is normalized per datum as the track count changes.
                    strength = smoothness*np.sqrt(len(observed)/200)
                    regularizer = strength*penalty
                    fit = lsq_linear(np.vstack([matrix,regularizer]),
                                      np.r_[observed,np.zeros(len(penalty))],
                                      bounds=(lower,upper),tol=1e-9)
                    if not fit.success:
                        continue
                    rmse = float(np.sqrt(np.mean((check@fit.x-target)**2)))
                    candidates.append(dict(knots=size,region=region,quadratic=quadratic,
                        smoothness=smoothness,theta=fit.x.tolist(),validation_rmse=rmse,
                        validation_samples=len(target),training_samples=len(observed)))
    if not candidates:
        raise RuntimeError('No valid inlet fit for the local track set')
    selected = min(candidates,key=lambda item:item['validation_rmse'])
    size = selected['knots']
    curve = spline_basis(heights,size)@np.array(selected['theta'][:size])
    return curve,selected,candidates


def predict_test(selected,test):
    matrix,observed = system(test,selected['knots'],VALIDATION_EDGE,selected['region'],selected['quadratic'])
    residual = matrix@np.asarray(selected['theta'])-observed
    return dict(samples=len(residual),axial_position_rmse=float(np.sqrt(np.mean(residual**2))))
