"""Factor the frozen SOAP operator at fixed observation coordinates.

Only the coordinate trunk is cached. The branch, lift, inverse objective and
all gradients still depend on the candidate inlet and Reynolds number.
"""
import jax
import jax.numpy as jnp
import numpy as np

from flow_operator.model import FREQUENCIES, context, mlp, point_fields, spline


class CachedObjective:
    def __init__(self, fitter, config):
        self.model = fitter.m
        model = self.model
        self.config = config
        params = model.params
        operator = model.operator.config
        if (operator.lift, operator.separate_pressure_trunk) != ('oseen', True):
            raise ValueError('Expected the separately trained pressure trunk and Oseen lift')

        def bases(xy):
            x, y = xy
            modes = jnp.array(FREQUENCIES)
            phase = jnp.pi*y*modes
            head = jnp.array([2*x/operator.length-1, 2*y-1])
            tail = jnp.array([*(jnp.exp(-x/s) for s in (.02,.05,.1,.3,1.,3.)),
                              jnp.log1p(20*x)/jnp.log1p(20*operator.length), x/operator.length])
            wave = jnp.concatenate([jnp.sin(phase), jnp.cos(phase)])
            psi = mlp(params['trunk'], jnp.concatenate([head, wave/jnp.tile(modes,2), tail]))
            pressure = mlp(params['pressure_trunk'], jnp.concatenate([head, wave, tail]))
            mask = 16*y*y*(1-y)**2*(-jnp.expm1(-x/.1))**2
            return jnp.stack([mask*psi, pressure])/jnp.sqrt(float(operator.latent))

        def lift(profile, re, xy):
            x, y = xy
            inlet, integral = spline(profile, y)
            base = 3*y*y-2*y**3
            k = jnp.pi*profile['effective_mode']
            alpha = 2*k*k/(jnp.sqrt(re*re+4*k*k)+re)
            delta = re*alpha/(k+alpha)
            decay = jnp.exp(-alpha*x)*(1-alpha*jnp.expm1(-delta*x)/delta)
            return jnp.array([base+(integral-base)*decay,
                              12/re*(operator.length-x)+(k+alpha)/re*(inlet-6*y*(1-y))*jnp.exp(-k*x)])

        basis_grad = jax.jacfwd(bases)
        lift_grad = jax.jacfwd(lift, argnums=2)

        def basis_velocity(xy):
            psi = basis_grad(xy)[0]
            return jnp.stack([psi[:,1], -psi[:,0]])

        def lift_velocity(profile, re, xy):
            psi = lift_grad(profile, re, xy)[0]
            return jnp.array([psi[1], -psi[0]])

        def basis_terms(xy):
            jac = basis_grad(xy)
            hess = jax.jacfwd(basis_grad)(xy)[0]
            third = jax.jacfwd(jax.jacfwd(basis_grad))(xy)[0]
            return (basis_velocity(xy), jnp.stack([hess[:,1,:],-hess[:,0,:]]),
                    jnp.stack([third[:,1,0,0]+third[:,1,1,1],-third[:,0,0,0]-third[:,0,1,1]]),
                    jac[1].T)

        def lift_terms(profile, re, xy):
            jac = lift_grad(profile,re,xy)
            hess = jax.jacfwd(lift_grad,argnums=2)(profile,re,xy)[0]
            third = jax.jacfwd(jax.jacfwd(lift_grad,argnums=2),argnums=2)(profile,re,xy)[0]
            return (jnp.array([jac[0,1],-jac[0,0]]), jnp.stack([hess[1,:],-hess[0,:]]),
                    jnp.array([third[1,0,0]+third[1,1,1],-third[0,0,0]-third[0,1,1]]),jac[1])

        probe = np.random.default_rng(71105).uniform([.001,.001],[4.999,.999],(config.physics_points,2))
        probe[:,0] = 5*(probe[:,0]/5)**2
        probes = jnp.asarray(probe)
        self.basis_velocity = jax.jit(jax.vmap(basis_velocity))
        probe_basis = jax.jit(jax.vmap(basis_terms))(probes) if config.physics_weight else None
        builder = model.builder

        def velocity(profile, re, branch, xy, spatial_basis):
            lift_data = jax.vmap(lift_velocity, in_axes=(None, None, 0))(profile, re, xy)
            return lift_data + spatial_basis @ branch[0]

        def momentum(profile, re, branch):
            vel, grad, lap, pressure = jax.vmap(lift_terms, in_axes=(None, None, 0))(profile, re, probes)
            bv, bg, bl, bp = probe_basis
            vel += bv @ branch[0]
            grad += jnp.einsum('ncld,l->ncd', bg, branch[0])
            lap += bl @ branch[0]
            pressure += bp @ branch[1]
            return jnp.einsum('ncd,nd->nc', grad, vel) + pressure - lap/re

        def positions(profile, speed, re, data, branch):
            lift_data = jax.vmap(jax.vmap(lift_velocity, in_axes=(None,None,0)), in_axes=(None,None,0))(profile,re,data['xy'])
            correction = jnp.einsum('ntcl,l->ntc',data['integrated_basis'],branch[0])
            return data['known'] + speed*(data['weights']@lift_data + correction)

        def loss(theta, data, count):
            z,speed,re = model.decode(theta)
            profile = builder.arrays(z)
            branch = context(params,profile,re,operator)
            predicted = positions(profile,speed,re,data,branch)
            residual = (predicted-data['observed'])*data['scales'][...,None]/config.position_scale
            aa,bb = profile['coeff'][0],profile['coeff'][1]
            curvature = jnp.sum((12*aa*aa+12*aa*bb+4*bb*bb)/builder.h**3)
            physical = 0.
            if probe_basis is not None:
                residual_momentum = momentum(profile, re, branch)
                physical = config.physics_weight*jnp.mean(jnp.sum((speed**2*residual_momentum/config.physics_scale)**2,axis=-1))
            regularization = config.smoothness*curvature+config.shape_prior*jnp.mean(z*z)+config.mean_prior*theta[-1]**2+physical
            dense = builder.dense@(jnp.asarray(builder.base)*jnp.exp(jnp.pad(z,(1,1))))
            return jnp.sum(residual**2)/count+regularization+1e5*jnp.mean(jnp.minimum(dense,0)**2)

        self.velocity = velocity
        self.momentum = momentum

        self.loss = jax.jit(jax.value_and_grad(loss))
        self.positions = positions
        self.point = jax.jit(lambda xy, branch, profile, re, speed:
                             speed*point_fields(params,branch,profile,re,xy,operator)[:2])

    def data(self, tracks, integral_observations):
        # Construct each trajectory's quadrature separately to avoid the dense
        # block matrix across all particles. Pre-integrate the frozen trunk.
        observations = [integral_observations([track],5,self.config.quadrature_order) for track in tracks]
        data = {key: jnp.asarray(np.stack([ob[key] for ob in observations]))
                for key in ('xy','weights','known','observed','scales')}
        shape = data['xy'].shape
        basis = self.basis_velocity(data['xy'].reshape(-1,2)).reshape(*shape[:2],2,-1)
        integrated = jnp.einsum('nts,nscl->ntcl',data['weights'],basis)
        return {**data, 'integrated_basis': integrated}
