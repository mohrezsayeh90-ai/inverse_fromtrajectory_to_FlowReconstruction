import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.signal import find_peaks

from flow_operator.profiles import Inlet,stack
from flow_operator.model import Config,initialize,make_functions,spline,context,point_fields
from flow_operator.checkpoint import save,load


@pytest.fixture(scope='module')
def setup_model():
    c=Config(width=12,latent=6,depth=2,encoder_width=12)
    p=jax.tree_util.tree_map(lambda z:z.astype(jnp.float64),initialize(c))
    return c,p,make_functions(c)


@pytest.mark.parametrize('kind',['nine','twenty_five','beta','flat','asymmetric'])
def test_full_input_is_preserved_at_boundary(kind,setup_model):
    c,p,f=setup_model;y=np.linspace(0,1,257)
    values={'nine':6*y*(1-y)*(1-.6*np.cos(18*np.pi*y)),
            'twenty_five':6*y*(1-y)*(1-.6*np.cos(50*np.pi*y)),
            'beta':630*y**4*(1-y)**4,
            'flat':(-np.expm1(-(y/.05)**2))*(-np.expm1(-((1-y)/.05)**2)),
            'asymmetric':y*(1-y)*np.exp(-((y-.3)/.15)**2)}[kind]
    inlet=Inlet.from_samples(y,values,65);data=stack([inlet],np.float64)
    dense=np.linspace(0,1,1025)
    xy=np.column_stack([np.zeros_like(dense),dense])[None]
    # Nonzero learned correction, and then a large parameter perturbation.
    for params in [p,jax.tree_util.tree_map(lambda a:a+.03,p)]:
        out=np.asarray(f['fields'](params,data,np.array([55.]),xy))[0]
        np.testing.assert_allclose(out[:,0],inlet.spline(dense),atol=1e-10)
        assert np.max(abs(out[:,1]))<1e-10
        walls=np.array([[[x,w] for w in [0.,1.] for x in [.01,1.,4.99]]])
        assert np.max(abs(np.asarray(f['fields'](params,data,np.array([55.]),walls))[...,:2]))<1e-9
    if kind=='nine':assert len(find_peaks(out[:,0])[0])==9
    if kind=='twenty_five':assert len(find_peaks(out[:,0])[0])==25


def test_spline_integral_and_spatial_derivatives(setup_model):
    y=np.sort(np.r_[0.,np.random.default_rng(9).uniform(size=40),1.])
    inlet=Inlet.from_samples(y,6*y*(1-y)*(1+.2*np.cos(18*np.pi*y)),65)
    a={k:jnp.array(v) for k,v in inlet.arrays(np.float64).items()}
    for z in [.1234,.4321,.8765]:
        f,I=spline(a,z)
        np.testing.assert_allclose(f,inlet.spline(z),atol=1e-10)
        np.testing.assert_allclose(I,inlet.spline.antiderivative()(z)-inlet.spline.antiderivative()(0),atol=1e-10)
        np.testing.assert_allclose(jax.grad(lambda t:spline(a,t)[1])(z),f,atol=1e-9)
        np.testing.assert_allclose(jax.grad(jax.grad(lambda t:spline(a,t)[0]))(z),inlet.spline(z,2),atol=1e-7)


def test_exact_poiseuille_and_outlet_and_mass(setup_model):
    c,p,f=setup_model
    zero=jax.tree_util.tree_map(jnp.zeros_like,p)
    y=np.linspace(0,1,65);inlet=Inlet.from_samples(y,6*y*(1-y),65)
    data=stack([inlet],np.float64)
    xy=np.array([[[.123,.234],[1.73,.712],[4.81,.323]]])
    out=np.asarray(f['fields'](zero,data,np.array([40.]),xy))[0]
    truth=np.column_stack([6*xy[0,:,1]*(1-xy[0,:,1]),np.zeros(3),12/40*(5-xy[0,:,0])])
    np.testing.assert_allclose(out,truth,atol=1e-9)
    assert np.max(abs(np.asarray(f['residual'](zero,data,np.array([40.]),xy))))<1e-8
    assert np.max(abs(np.asarray(f['outlet'](zero,data,np.array([40.]),np.array([[.13,.37,.81]])))))<1e-9
    r=np.asarray(f['residual'](p,data,np.array([40.]),xy))
    assert np.max(abs(r[...,2]))<1e-10


def test_residual_against_independent_finite_differences(setup_model):
    c,p,f=setup_model;y=np.linspace(0,1,257)
    inlet=Inlet.from_samples(y,6*y*(1-y)*(1+.3*np.cos(18*np.pi*y)),65)
    a={k:jnp.array(v) for k,v in inlet.arrays(np.float64).items()}
    re=55.;b=context(p,a,re,c);z=np.array([.071,.3213]);h=1e-5
    fn=lambda q:np.asarray(point_fields(p,b,a,re,jnp.array(q),c))
    value=fn(z);u,v,_=value
    jac=np.column_stack([(fn(z+np.eye(2)[k]*h)-fn(z-np.eye(2)[k]*h))/(2*h) for k in range(2)])
    lap=sum((fn(z+np.eye(2)[k]*h)-2*value+fn(z-np.eye(2)[k]*h))/h**2 for k in range(2))
    finite=np.array([u*jac[0,0]+v*jac[0,1]+jac[2,0]-lap[0]/re,
                     u*jac[1,0]+v*jac[1,1]+jac[2,1]-lap[1]/re,jac[0,0]+jac[1,1]])
    actual=np.asarray(f['residual'](p,stack([inlet],np.float64),np.array([re]),z[None,None]))[0,0]
    np.testing.assert_allclose(actual,finite,rtol=2e-4,atol=2e-4)


def test_encoder_accepts_different_quadrature_and_input_sizes(setup_model):
    c,p,f=setup_model
    for n,q in [(65,33),(257,129),(513,257)]:
        y=np.linspace(0,1,n);inlet=Inlet.from_samples(y,630*y**4*(1-y)**4,q)
        out=f['fields'](p,stack([inlet],np.float64),np.array([55.]),np.array([[[.31,.43]]]))
        assert np.isfinite(np.asarray(out)).all()


def test_checkpoint_roundtrip_and_resume_state(tmp_path,setup_model):
    c,p,f=setup_model
    rng=np.random.default_rng(31);rng.uniform(size=17)
    path=tmp_path/'checkpoint.npz'
    save(path,p,c,7,rng.bit_generator.state,(p,p),dict(validation_status='not_accepted'))
    got,config,meta,state=load(path,optimizer=True)
    assert config==c and meta['step']==7
    for a,b in zip(jax.tree_util.tree_leaves(p),jax.tree_util.tree_leaves(got)):np.testing.assert_array_equal(a,b)
    other=np.random.default_rng();other.bit_generator.state=meta['rng_state']
    np.testing.assert_array_equal(other.uniform(size=10),rng.uniform(size=10))
