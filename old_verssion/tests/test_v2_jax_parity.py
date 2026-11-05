"""Cross-framework check of fields AND third-coordinate-derivative NS residuals."""
import numpy as np
import pytest
import torch
jax=pytest.importorskip('jax')
import jax.numpy as jnp
from channel_operator.v2_model import StreamDeepONet,physics
from channel_operator.v2_profiles import inlet_values
from channel_operator.v2_jax import make_functions


@pytest.mark.parametrize('pressure_mode',['viscous','convective'])
def test_jax_matches_torch_fields_third_derivatives_and_parameter_gradients(pressure_mode):
    torch.set_num_threads(2);torch.manual_seed(31)
    model=StreamDeepONet(width=8,latent=4,depth=2,pressure_mode=pressure_mode)
    profiles=inlet_values(np.array([[.1,-.1,.05,.3],[0.,0.,0.,.6]])).astype(np.float32)
    re=np.array([10.,100.],dtype=np.float32)
    xy=np.array([[[.17,.217],[.91,.731]],[[.2,.213],[1.13,.627]]],dtype=np.float32)
    params={k:jnp.asarray(v.detach().numpy()) for k,v in model.named_parameters()}
    f=make_functions(latent=4,depth=2,pressure_mode=pressure_mode)
    tt=lambda z:torch.tensor(z,dtype=torch.float32)
    expected=model(tt(profiles),tt(re),tt(xy)).detach().numpy()
    actual=np.asarray(f['fields'](params,profiles,re,xy))
    np.testing.assert_allclose(actual,expected,rtol=2e-4,atol=2e-5)
    torch_res=physics(model,tt(profiles),tt(re),tt(xy))
    expected=np.stack([v.detach().numpy() for v in torch_res],-1)
    actual=np.asarray(f['physics'](params,profiles,re,xy))
    np.testing.assert_allclose(actual,expected,rtol=3e-4,atol=2e-3)
    model.zero_grad();sum((v*v).sum() for v in torch_res[:2]).backward()
    grad=jax.grad(lambda theta:jnp.sum(f['physics'](theta,profiles,re,xy)[...,:2]**2))(params)
    a=np.concatenate([np.asarray(grad[k]).ravel() for k,_ in model.named_parameters()])
    b=np.concatenate([v.grad.numpy().ravel() for _,v in model.named_parameters()])
    assert np.isfinite(a).all()
    assert np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))>.9999
