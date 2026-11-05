import torch
from torch import nn
from .profiles import SENSORS, Y, SPLINE_MAP


def mlp(dim, width, output, layers=3):
    blocks = [nn.Linear(dim, width), nn.Tanh()]
    for _ in range(layers-1):
        blocks += [nn.Linear(width, width), nn.Tanh()]
    blocks += [nn.Linear(width, output)]
    return nn.Sequential(*blocks)


class ChannelDeepONet(nn.Module):
    """Branch: inlet samples + log(Re). Trunk: nondimensional x,y.

    Hard inlet/no-slip constraints; Poiseuille is a baseline, not training data.
    Input shapes: profiles [B,33], Re [B], coordinates [B,N,2].
    """
    def __init__(self, length=5., width=64, latent=32, entrance_features=False):
        super().__init__()
        self.config = dict(length=length, width=width, latent=latent)
        self.entrance_features = entrance_features
        if entrance_features:
            self.config["entrance_features"] = True
        self.length, self.latent = length, latent
        self.branch = mlp(SENSORS+1, width, 3*latent)
        self.trunk = mlp(14 if entrance_features else 10, width, 3*latent)
        nn.init.normal_(self.trunk[-1].weight, std=1e-3)
        nn.init.zeros_(self.trunk[-1].bias)
        self.register_buffer("spline_map", torch.tensor(SPLINE_MAP.copy(), dtype=torch.float32))
        self.register_buffer("base", torch.tensor(6*Y*(1-Y), dtype=torch.float32))

    def inlet(self, profiles, y):
        # All spline polynomial coefficients depend linearly on branch sensors.
        coeff = torch.einsum("kjs,bs->bkj", self.spline_map, profiles)
        index = (y.detach()*(SENSORS-1)).long().clamp(0, SENSORS-2)
        t = y - index / (SENSORS-1)
        c = coeff.gather(2, index[:, None, :].expand(-1, 4, -1))
        return ((c[:, 0]*t+c[:, 1])*t+c[:, 2])*t+c[:, 3]

    def inlet_integral(self, profiles, y):
        coeff = torch.einsum("kjs,bs->bkj", self.spline_map, profiles)
        h = 1/(SENSORS-1)
        integrals = coeff[:, 0]*h**4/4 + coeff[:, 1]*h**3/3 + coeff[:, 2]*h*h/2 + coeff[:, 3]*h
        prefix = torch.cat([integrals.new_zeros(len(profiles), 1), integrals.cumsum(-1)], -1)
        index = (y.detach()*(SENSORS-1)).long().clamp(0, SENSORS-2)
        t = y-index*h
        c = coeff.gather(2, index[:, None, :].expand(-1, 4, -1))
        return prefix.gather(1, index) + c[:, 0]*t**4/4+c[:, 1]*t**3/3+c[:, 2]*t*t/2+c[:, 3]*t

    def forward(self, profiles, reynolds, xy):
        x, y = xy[..., 0], xy[..., 1]
        condition = torch.cat([profiles-self.base, (torch.log(reynolds/10)/torch.log(reynolds.new_tensor(10.)))[:, None]], -1)
        b = self.branch(condition).reshape(-1, 3, self.latent)
        coords = torch.stack((2*x/self.length-1, 2*y-1), -1)
        modes = torch.arange(1, 5, device=xy.device, dtype=xy.dtype)
        phase = torch.pi*y[..., None]*modes
        features = torch.cat([coords, torch.sin(phase), torch.cos(phase)], -1)
        if self.entrance_features:
            # Resolve short entrance scales without asking tanh(x/L) to invent them.
            local = torch.stack([torch.log1p(20*x)/torch.log(x.new_tensor(101.)),
                                 torch.exp(-x), torch.exp(-4*x), torch.exp(-16*x)], -1)
            features = torch.cat([features, local], -1)
        t = self.trunk(features).reshape(*x.shape, 3, self.latent)
        raw = torch.einsum("bol,bnol->bno", b, t) / self.latent**.5
        axial_mask = 1-torch.exp(-x) if self.entrance_features else x/self.length
        mask = axial_mask*4*y*(1-y)
        # Divergence-free entrance-decay baseline; the network learns corrections.
        ell = .05*reynolds[:, None]+.1
        decay = torch.exp(-(x/ell)**2)
        delta = self.inlet(profiles, y)-6*y*(1-y)
        psi_delta = self.inlet_integral(profiles, y)-(3*y*y-2*y**3)
        u = 6*y*(1-y)+delta*decay + mask*raw[..., 0]
        v = psi_delta*(2*x/ell**2)*decay + mask*raw[..., 1]
        p = (12*(self.length-x) + 12*raw[..., 2]) / reynolds[:, None]
        return torch.stack((u, v, p), -1)


def gradient(value, xy):
    return torch.autograd.grad(value.sum(), xy, create_graph=True)[0]


def residuals(model, profiles, reynolds, xy):
    xy = xy.detach().requires_grad_(True)
    fields = model(profiles, reynolds, xy)
    u, v, p = fields.unbind(-1)
    gu, gv, gp = [gradient(z, xy) for z in (u, v, p)]
    lapu = gradient(gu[..., 0], xy)[..., 0] + gradient(gu[..., 1], xy)[..., 1]
    lapv = gradient(gv[..., 0], xy)[..., 0] + gradient(gv[..., 1], xy)[..., 1]
    nu = 1/reynolds[:, None]
    mass = gu[..., 0]+gv[..., 1]
    mx = u*gu[..., 0]+v*gu[..., 1]+gp[..., 0]-nu*lapu
    my = u*gv[..., 0]+v*gv[..., 1]+gp[..., 1]-nu*lapv
    return mass, mx, my


def outlet_residuals(model, profiles, reynolds, xy):
    xy = xy.detach().requires_grad_(True)
    fields = model(profiles, reynolds, xy)
    # Standard unsymmetric-gradient do-nothing boundary from the FEM weak form.
    ux = gradient(fields[..., 0], xy)[..., 0]
    vx = gradient(fields[..., 1], xy)[..., 0]
    return ux-reynolds[:, None]*fields[..., 2], vx


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = ChannelDeepONet(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
