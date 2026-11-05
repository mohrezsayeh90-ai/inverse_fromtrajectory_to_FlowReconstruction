import numpy as np
import pytest
import torch
from channel_operator.model import ChannelDeepONet, residuals, outlet_residuals, gradient
from channel_operator.profiles import from_coefficients, sample_profiles, Y, normalize, load_csv, INTEGRAL_WEIGHTS
from channel_operator.reference import solve_channel


@pytest.mark.parametrize("entrance_features", [False, True])
def test_hard_boundary_conditions_and_integral(entrance_features):
    model = ChannelDeepONet(width=16, latent=8, entrance_features=entrance_features).double()
    profiles, _ = sample_profiles(np.random.default_rng(1), 3)
    p = torch.tensor(profiles)
    r = torch.tensor([10., 50., 100.], dtype=torch.float64)
    y = torch.linspace(0, 1, 33, dtype=torch.float64).repeat(3, 1)
    xy = torch.stack([torch.zeros_like(y), y], -1)
    result = model(p, r, xy)
    assert torch.max(torch.abs(result[..., 0]-p)) < 1e-6
    assert torch.max(torch.abs(result[..., 1])) < 1e-12
    for wall in [0., 1.]:
        xy = torch.stack([5*y, torch.full_like(y, wall)], -1)
        assert torch.max(torch.abs(model(p, r, xy)[..., :2])) < 2e-6
    assert torch.max(torch.abs(model.inlet_integral(p, torch.ones_like(y))-1)) < 1e-6


def test_poiseuille_pde_and_outlet_residuals():
    model = ChannelDeepONet(width=16, latent=8).double()
    # Zero correction tests equations and nondimensionalization, not learning.
    with torch.no_grad():
        model.trunk[-1].weight.zero_()
        model.trunk[-1].bias.zero_()
    p = torch.tensor(from_coefficients(np.zeros((3, 4))))
    r = torch.tensor([10., 40., 100.], dtype=torch.float64)
    xy = torch.rand(3, 64, 2, dtype=torch.float64)
    xy[..., 0] *= 5
    for residual in residuals(model, p, r, xy):
        assert residual.abs().max() < 2e-4
    xy[..., 0] = 5
    for residual in outlet_residuals(model, p, r, xy):
        assert residual.abs().max() < 2e-4


def test_physics_loss_backpropagates_to_branch():
    model = ChannelDeepONet(width=16, latent=8)
    p = torch.tensor(from_coefficients(np.array([[.2, -.1, .05, 0.]])), dtype=torch.float32)
    xy = torch.rand(1, 32, 2)
    xy[..., 0] *= 5
    loss = sum(r.square().mean() for r in residuals(model, p, torch.tensor([40.]), xy))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.branch[0].weight.grad.abs().sum() > 0


def test_profile_constraints():
    with pytest.raises(ValueError):
        normalize(np.ones(33))
    with pytest.raises(ValueError):
        normalize(np.zeros(33))
    with pytest.raises(ValueError):
        from_coefficients([5., 0., 0., 0.])


def test_custom_csv_preserves_shape_and_normalizes_mean(tmp_path):
    expected = from_coefficients([.25, -.1, .05, 0.])
    path = tmp_path/"inlet.csv"
    np.savetxt(path, np.column_stack([Y, 2.7*expected]), delimiter=",", header="y,u", comments="")
    profile = load_csv(path)
    assert np.max(np.abs(profile-expected)) < 1e-12
    assert abs(profile@INTEGRAL_WEIGHTS-1) < 1e-12


@pytest.mark.parametrize('graded', [False, True])
def test_independent_fem_poiseuille(graded):
    xy = np.array([[.1, .1], [1.5, .3], [4.9, .7]])
    mesh = dict(x_nodes=5*np.linspace(0, 1, 17)**2) if graded else {}
    evaluate, info = solve_channel(from_coefficients(np.zeros(4)), 40., nx=16, ny=6, **mesh)
    truth = np.column_stack([6*xy[:, 1]*(1-xy[:, 1]), np.zeros(3), 12/40*(5-xy[:, 0])])
    assert np.max(np.abs(evaluate(xy)-truth)) < 1e-9
    assert info["nonlinear_residual"] < 1e-9
