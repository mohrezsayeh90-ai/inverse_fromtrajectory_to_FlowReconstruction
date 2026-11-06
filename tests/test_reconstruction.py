import json
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flow_reconstruction.particles import Track,simulate,integral_observations
from flow_reconstruction.profile import ProfileBuilder
from flow_reconstruction.inverse import FlowReconstructor,InverseConfig
from flow_operator.model import Config,initialize
from flow_operator.checkpoint import save


def test_particle_constant_flow_exact_and_opposite_launch():
    t=np.linspace(0,.6,51);u=np.array([1.2,-.1]);q=np.array([3.,.3]);v=np.array([-2.,1.]);tau=.12
    solved=simulate(lambda q:u,t,q,v,tau)
    exact=q+u*t[:,None]+tau*(v-u)*(-np.expm1(-t[:,None]/tau))
    assert solved['completed'];np.testing.assert_allclose(solved['position'],exact,atol=2e-9)
    assert solved['velocity'][-1,0]>0


def test_integral_identity_uses_positions_and_known_launch_only():
    t=np.r_[0.,.017,.051,.14,.23];u=np.array([.8,.1]);v=np.array([3.,-1.]);tau=.12;q0=np.array([.1,.8])
    q=q0+u*t[:,None]+tau*(v-u)*(-np.expm1(-t[:,None]/tau))
    track=Track('known',t,q,v,tau);d=integral_observations([track],order=8)
    prediction=d['known']+d['weights']@np.tile(u,(len(d['xy']),1))
    np.testing.assert_allclose(prediction,q[1:],atol=1e-12)


def test_boundary_termination_and_invalid_times():
    s=simulate(lambda q:np.array([1.,0.]),np.linspace(0,1,31),[4.9,.5],[1.,0.],.1)
    assert not s['completed'] and abs(s['boundary_exit_time']-.1)<1e-8
    with pytest.raises(ValueError,match='increase'):Track('bad',[0,.1,.1,.3],np.zeros((4,2)),[1,0],.1)


def test_exact_corner_inward_launch_does_not_terminate_at_initial_time():
    for q,v in [([0.,1.],[1.,-1.]),([5.,0.],[-1.,1.])]:
        s=simulate(lambda q:np.zeros(2),np.linspace(0,.1,6),q,v,.1)
        assert s['completed'] and s['boundary_exit_time'] is None
    with pytest.raises(ValueError,match='into'):simulate(lambda q:np.zeros(2),[0,.1],[0,1],[-1,-1],.1)


def test_differentiable_complete_spline_matches_original_interface():
    b=ProfileBuilder(65);z=.4*np.cos(18*np.pi*b.y[1:-1])+.2*np.sin(3*np.pi*b.y[1:-1])
    actual=b.arrays(jnp.array(z));reference=b.inlet(z).arrays(np.float64)
    for key in reference:np.testing.assert_allclose(actual[key],reference[key],rtol=1e-9,atol=1e-9)
    gradient=jax.grad(lambda z:jnp.sum(b.arrays(z)['encoder_f']**2))(jnp.array(z))
    assert np.isfinite(gradient).all() and np.linalg.norm(gradient)>0


def make_small_inverse(tmp_path,iterations=3,physics_weight=0):
    cfg=Config(width=8,latent=4,depth=1,separate_pressure_trunk=True)
    path=tmp_path/'forward.npz';params=jax.tree_util.tree_map(jnp.zeros_like,initialize(cfg))
    save(path,params,cfg,1,np.random.default_rng(7).bit_generator.state)
    return FlowReconstructor(path,InverseConfig(reference_reynolds=40.,profile_points=9,starts=1,
        max_iterations=iterations,mean_speed_bounds=(.8,1.3),smoothness=.01,shape_prior=.1,position_scale=.003,physics_weight=physics_weight,physics_points=16),allow_unvalidated=True)


def analytic_tracks(speed=1.15):
    tracks=[];t=np.linspace(0,.4,41)
    for name,q,v in [('a',[.03,.97],[6.,-4.]),('b',[4.97,.03],[-6.,4.])]:
        sim=simulate(lambda q:np.array([speed*6*q[1]*(1-q[1]),0.]),t,q,v,.12)
        assert sim['completed'];tracks.append(Track(name,t,sim['position'],v,.12))
    return tracks


def test_inverse_objective_gradient_against_finite_differences(tmp_path):
    model=make_small_inverse(tmp_path).prepare(analytic_tracks());x=np.array([.1,-.05,.02,.03,-.02,.07,.05])
    transform=np.asarray(model.shape_transform)
    assert np.linalg.matrix_rank(transform)==transform.shape[0]
    assert np.linalg.eigvalsh(transform).min()>0
    assert float(model.decode(jnp.asarray(x))[0][model.anchor])==0
    value,grad=model.loss_and_gradient(jnp.array(x))
    for k in [0,3,6]:
        h=1e-5;step=np.eye(len(x))[k]*h
        finite=(float(model.loss_and_gradient(x+step)[0])-float(model.loss_and_gradient(x-step)[0]))/(2*h)
        np.testing.assert_allclose(grad[k],finite,rtol=2e-4,atol=2e-5)


def test_inverse_recovers_known_mean_and_preserves_checkpoint(tmp_path):
    model=make_small_inverse(tmp_path,iterations=100);result=model.fit(analytic_tracks())
    assert abs(result['report']['mean_speed']-1.15)<.025
    assert result['report']['forward_weights_unchanged']
    assert result['report']['trajectory_consistency_passed']
    assert not result['report']['forward_operator_benchmark_accepted']
    assert not result['report']['unique_flow_guaranteed']


def test_inverse_rejects_reynolds_extrapolation():
    with pytest.raises(ValueError,match='inside'):InverseConfig(reference_reynolds=100.,mean_speed_bounds=(.5,1.5))


def test_physics_regularizer_gradient_and_saved_fit_roundtrip(tmp_path):
    model=make_small_inverse(tmp_path,physics_weight=.02).prepare(analytic_tracks())
    x=np.array([.1,-.05,.02,.03,-.02,.07,.05]);value,gradient=model.loss_and_gradient(x)
    step=np.eye(len(x))[1]*1e-5
    finite=(float(model.loss_and_gradient(x+step)[0])-float(model.loss_and_gradient(x-step)[0]))/2e-5
    np.testing.assert_allclose(gradient[1],finite,rtol=3e-4,atol=2e-5)
    from flow_reconstruction.io import load_fit,inverse_signature
    from flow_operator.provenance import runtime_signature
    path=tmp_path/'fit.json';path.write_text(json.dumps(dict(format='stokes_trajectory_inverse_v1',
        checkpoint_sha256=model.operator.sha256,runtime_signature=runtime_signature(),inverse_runtime_signature=inverse_signature(),
        config=__import__('dataclasses').asdict(model.config),theta=x.tolist())))
    loaded,theta=load_fit(path,model.operator.path,allow_unvalidated=True)
    xy=np.array([[.1,.3],[3.,.6]])
    np.testing.assert_allclose(loaded.fields(theta,xy),model.fields(x,xy),atol=1e-12)
    data=json.loads(path.read_text());data['checkpoint_sha256']='wrong';path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='differs'):load_fit(path,model.operator.path,allow_unvalidated=True)
