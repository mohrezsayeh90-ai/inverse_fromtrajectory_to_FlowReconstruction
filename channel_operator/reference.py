"""Independent Taylor-Hood P2/P1 steady Navier-Stokes finite-element solver.

No PINN predictions are used by this solver. Natural outlet uses viscosity*du/dn-pn=0.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.sparse import bmat, csr_matrix
from skfem import MeshTri, Basis, ElementTriP2, ElementTriP1, ElementVector, BilinearForm, asm, condense, solve
from skfem.helpers import ddot, dot, div
from .profiles import Y


@BilinearForm
def diffusion(u, v, w):
    return ddot(u.grad, v.grad)


@BilinearForm
def divergence(u, q, w):
    return -q*div(u)


@BilinearForm
def convection(u, v, w):
    return dot(np.einsum("j...,ij...->i...", w.wind, u.grad), v)


def solve_channel(profile, reynolds, length=5., nx=40, ny=16, tolerance=1e-7, max_iterations=150, inlet_function=None):
    mesh = MeshTri.init_tensor(np.linspace(0, length, nx+1), np.linspace(0, 1, ny+1))
    ub = Basis(mesh, ElementVector(ElementTriP2()), intorder=5)
    pb = Basis(mesh, ElementTriP1(), intorder=5)
    a, b = asm(diffusion, ub)/reynolds, asm(divergence, ub, pb)
    walls = ub.get_dofs(lambda x: np.isclose(x[1], 0) | np.isclose(x[1], 1))
    inlet = ub.get_dofs(lambda x: np.isclose(x[0], 0))
    fixed = np.union1d(walls.all(), inlet.all())
    bc = np.zeros(ub.N+pb.N)
    inlet_u = inlet.all(["u^1"])
    inlet_function = CubicSpline(Y, profile) if inlet_function is None else inlet_function
    bc[inlet_u] = inlet_function(ub.doflocs[1, inlet_u])
    zero = csr_matrix((pb.N, pb.N))
    stokes = bmat([[a, b.T], [b, zero]], format="csr")
    solution = solve(*condense(stokes, np.zeros(len(bc)), x=bc, D=fixed))
    free = np.setdiff1d(np.arange(len(bc)), fixed)
    converged = False
    for iteration in range(max_iterations):
        c = asm(convection, ub, wind=ub.interpolate(solution[:ub.N]))
        matrix = bmat([[a+c, b.T], [b, zero]], format="csr")
        updated = solve(*condense(matrix, np.zeros(len(bc)), x=bc, D=fixed))
        error = np.linalg.norm(updated[:ub.N]-solution[:ub.N]) / max(np.linalg.norm(updated[:ub.N]), 1e-12)
        solution = .7*updated + .3*solution
        solution[fixed] = bc[fixed]
        if error < tolerance:
            converged = True
            break
    # Check the nonlinear algebraic equations, not only the iterate increment.
    c = asm(convection, ub, wind=ub.interpolate(solution[:ub.N]))
    matrix = bmat([[a+c, b.T], [b, zero]], format="csr")
    nonlinear_residual = np.linalg.norm((matrix@solution)[free])
    if not converged or not np.isfinite(solution).all() or nonlinear_residual > 1e-6:
        raise RuntimeError(f"FEM failed at Re={reynolds}: delta={error:.3g}, residual={nonlinear_residual:.3g}")

    def evaluate(xy):
        xy = np.asarray(xy)
        velocity = ub.interpolator(solution[:ub.N])(xy.T).T
        pressure = pb.interpolator(solution[ub.N:])(xy.T)
        return np.column_stack([velocity, pressure])

    return evaluate, dict(iterations=iteration+1, increment=float(error), nonlinear_residual=float(nonlinear_residual), nx=nx, ny=ny)
