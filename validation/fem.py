"""Independent Taylor-Hood FEM for validation of the same steady NS/traction BVP.

The reference is a classical solver. It never calls the learned model.
"""
import numpy as np
from scipy.sparse import bmat,csr_matrix
from skfem import MeshTri,Basis,ElementTriP2,ElementTriP1,ElementVector,BilinearForm,asm,condense,solve
from skfem.helpers import ddot,dot,div


@BilinearForm
def diffusion(u,v,w):return ddot(u.grad,v.grad)


@BilinearForm
def divergence(u,q,w):return -q*div(u)


@BilinearForm
def convection(u,v,w):return dot(np.einsum('j...,ij...->i...',w.wind,u.grad),v)


def solve_reference(inlet,re,nx=80,ny=64):
    mesh=MeshTri.init_tensor(5*np.linspace(0,1,nx+1)**2,np.linspace(0,1,ny+1))
    ub=Basis(mesh,ElementVector(ElementTriP2()),intorder=5);pb=Basis(mesh,ElementTriP1(),intorder=5)
    a,b=asm(diffusion,ub)/re,asm(divergence,ub,pb)
    walls=ub.get_dofs(lambda x:np.isclose(x[1],0)|np.isclose(x[1],1))
    entrance=ub.get_dofs(lambda x:np.isclose(x[0],0))
    fixed=np.union1d(walls.all(),entrance.all());bc=np.zeros(ub.N+pb.N)
    ids=entrance.all(['u^1']);bc[ids]=inlet.spline(ub.doflocs[1,ids])
    zero=csr_matrix((pb.N,pb.N));matrix=bmat([[a,b.T],[b,zero]],format='csr')
    solution=solve(*condense(matrix,np.zeros_like(bc),x=bc,D=fixed))
    free=np.setdiff1d(np.arange(len(bc)),fixed)
    for i in range(150):
        c=asm(convection,ub,wind=ub.interpolate(solution[:ub.N]))
        matrix=bmat([[a+c,b.T],[b,zero]],format='csr')
        updated=solve(*condense(matrix,np.zeros_like(bc),x=bc,D=fixed))
        change=np.linalg.norm(updated[:ub.N]-solution[:ub.N])/max(np.linalg.norm(updated[:ub.N]),1e-12)
        solution=.7*updated+.3*solution;solution[fixed]=bc[fixed]
        if change<1e-7:break
    c=asm(convection,ub,wind=ub.interpolate(solution[:ub.N]))
    matrix=bmat([[a+c,b.T],[b,zero]],format='csr');residual=float(np.linalg.norm((matrix@solution)[free]))
    if change>=1e-7 or residual>1e-6 or not np.isfinite(solution).all():raise RuntimeError('Reference failed to converge')
    def evaluate(xy):
        xy=np.asarray(xy)
        return np.column_stack([ub.interpolator(solution[:ub.N])(xy.T).T,pb.interpolator(solution[ub.N:])(xy.T)])
    return evaluate,dict(nx=nx,ny=ny,iterations=i+1,nonlinear_residual=residual)
