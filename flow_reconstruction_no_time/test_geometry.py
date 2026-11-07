"""Checks specifically guarding time leakage, geometry and differentiable clocks."""
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import pytest
from .geometry import Curve,resample_curve,curve_metrics
from .inverse import positive_clock,integral_positions,GeometryConfig

def test_invalid_clock_and_boundary_launch():
    with pytest.raises(ValueError):GeometryConfig(clock_knots=2)
    with pytest.raises(ValueError):GeometryConfig(clock_smoothness=-1.)
    with pytest.raises(ValueError):resample_curve(np.zeros((4,2)),count=2)
    with pytest.raises(ValueError,match='into the channel'):
        Curve('bad',np.array([[0,0],[1,.2],[2,.3],[3,.4]]),[-1,1],.1)

def test_reject_timestamp_column(tmp_path):
    p=tmp_path/'bad.csv';p.write_text('t,x,y\n0,0,0\n1,1,.1\n2,2,.2\n3,3,.3\n')
    with pytest.raises(ValueError,match='forbidden'):Curve.read(p,'p',[1,1],.1)

def test_uniform_arclength_erases_sampling_density_and_retains_turn():
    base=np.array([[0,0],[1,.2],[.5,.7],[2,1.]])
    dense=np.concatenate([a[None]+np.linspace(0,1,n,endpoint=False)[:,None]*(b-a)[None] for a,b,n in zip(base[:-1],base[1:],[71,5,19])]+[base[-1:]])
    np.testing.assert_allclose(resample_curve(base)[0],resample_curve(dense)[0],atol=2e-14)
    assert np.any(np.diff(resample_curve(base)[0][:,0])<0)

def test_metrics_penalize_missing_branch_and_wrong_order():
    q=np.array([[0,0],[1,.2],[.5,.7],[2,1.]])
    m=curve_metrics(q,q[::-1],count=101)
    assert m['symmetric_distance_rmse_H']<1e-12 and m['ordered_arclength_rmse_H']>.5
    partial=np.vstack([q[:2],.5*(q[1]+q[2]),q[2]])
    assert curve_metrics(q,partial,count=101)['symmetric_hausdorff_H']>.5

def test_clock_positive_no_observed_duration_and_gradient():
    basis=jnp.asarray(np.column_stack([1-np.linspace(0,1,9),np.linspace(0,1,9)]))
    fn=lambda p:positive_clock(p,basis,.1,6.)
    np.testing.assert_allclose(fn(jnp.array([0.])),np.arange(9)*.1/6)
    assert np.all(np.diff(fn(jnp.array([100.])))>0)
    point=jnp.array([.5]);ad=np.asarray(jax.jacrev(fn)(point));fd=(np.asarray(fn(point+1e-5))-np.asarray(fn(point-1e-5)))/2e-5
    np.testing.assert_allclose(ad[:,0],fd,rtol=1e-8,atol=1e-10)

def test_integral_identity_against_exact_constant_flow():
    z,w=np.polynomial.legendre.leggauss(8);z=(z+1)/2;w/=2
    t=np.linspace(0,1,65);u=np.array([1.2,.13]);v0=np.array([6.,-1.2]);tau=.42;q0=np.array([0.,1.])
    fluid=np.broadcast_to(u,((len(t)-1)*len(z),2))
    pred=integral_positions(jnp.asarray(fluid),jnp.asarray(t),q0,v0,tau,jnp.asarray(z),jnp.asarray(w))
    exact=q0+u*t[1:,None]+tau*(v0-u)*(1-np.exp(-t[1:,None]/tau))
    np.testing.assert_allclose(pred,exact,atol=2e-14)
