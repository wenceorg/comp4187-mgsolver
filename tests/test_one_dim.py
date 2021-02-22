from functools import cached_property

import numpy
import pytest
from mgsolver import AbstractOperator, Grid1D, GridHierarchy, MGSolver, PETSc


@pytest.fixture
def coarse_grid():
    return Grid1D(11)


class Poisson3pt(AbstractOperator):
    def __init__(self, grid):
        super().__init__(grid)
        self.xloc = self.grid.createLocalVector()

    @cached_property
    def diagonal(self):
        D = self.grid.createGlobalVector()
        mx, = self.grid.getSizes()
        Hx = 1/(mx - 1)
        D.array[:] = 2 / Hx
        return D

    def mult(self, x, y):
        grid = self.grid
        xloc = self.xloc
        grid.globalToLocal(x, xloc, addv=PETSc.InsertMode.INSERT_VALUES)
        mx, = grid.getSizes()
        Hx = 1/(mx - 1)
        (xstart, ), (nx, ) = grid.getCorners()
        (xstartloc, ), (nxloc, ) = grid.getGhostCorners()
        xoff = xstart - xstartloc
        yarray = y.array.reshape(nx)
        xarray = xloc.array_r.reshape(nxloc)
        for i in range(nx):
            # Global number to figure out if we're on the boundary of
            # the domain.
            iglob = i + xstart
            # Diagonal application
            # Translate index in global vector to index in local
            # vector for xarray.
            iloc = i + xoff
            yarray[i] = 2 / Hx * xarray[iloc]
            if iglob == 0 or iglob == mx - 1:
                # On boundary, nothing else to do
                pass
            else:
                iloc = i + xoff - 1
                yarray[i] += -1/Hx * xarray[iloc]
                iloc = i + xoff + 1
                yarray[i] += -1/Hx * xarray[iloc]

    def as_sparse_matrix(self):
        grid = self.grid
        A = grid.createMatrix()
        mx, = grid.getSizes()
        Hx = 1/(mx - 1)
        (xstart, ), (nx, ) = grid.getCorners()
        row = PETSc.Mat.Stencil()
        col = PETSc.Mat.Stencil()
        for i in range(nx):
            iglob = i + xstart
            # Insertion into matrice is with global numbers
            row.i = iglob
            col.i = iglob
            value = 2 / Hx
            A.setValueStencil(row, col, value,
                              addv=PETSc.InsertMode.INSERT_VALUES)
            if iglob == 0 or iglob == mx - 1:
                # On boundary, nothing else to do
                pass
            else:
                col.i = iglob - 1
                value = -1/Hx
                A.setValueStencil(row, col, value,
                                  addv=PETSc.InsertMode.INSERT_VALUES)
                col.i = iglob + 1
                A.setValueStencil(row, col, value,
                                  addv=PETSc.InsertMode.INSERT_VALUES)
        A.assemblyBegin(A.AssemblyType.FINAL_ASSEMBLY)
        A.assemblyEnd(A.AssemblyType.FINAL_ASSEMBLY)
        return A


def rhs(grid):
    # Differentiating twice, we find that the right hand side forcing
    # term should be f(x) = pi**2 / 4 * sin(pi/2 x)
    # With the boundary conditions that u = 0 for x = 0
    # and u = 1 for x = 1
    # Since we put a scaled identity on the diagonal for the boundary
    # nodes, we therefore just need to ensure that the right hand side
    # has an appropriate scaling too.
    b = grid.createGlobalVector()
    mx, = grid.getSizes()
    (xstart, ), (nx, ) = grid.getCorners()
    Hx = 1/(mx - 1)
    barray = b.array.reshape(nx)
    coords = grid.getCoordinates().array_r.reshape(nx, 1)
    for i in range(nx):
        x = coords[i, 0]
        if x == 0:
            barray[i] = 0
        elif x == 1:
            barray[i] = 2 / Hx
        else:
            barray[i] = numpy.pi**2/4 * numpy.sin(numpy.pi/2 * x) * Hx
    return b, numpy.sin(numpy.pi * coords[:, 0] / 2)


def run_solve(hierarchy):
    fine_grid = hierarchy[-1]
    operator = Poisson3pt

    u = fine_grid.createGlobalVector()
    b, expect = rhs(fine_grid)

    # We will solve -\nabla^2 u = f(x)
    # Such that the exact solution is u = sin(pi/2 x)
    solver = MGSolver(hierarchy, operator)
    # This alpha is good for 1D, but maybe not others
    solver.solve(u, b, maxiter=100, alpha=0.8, rtol=1e-8, monitor=False)

    # Return L2 error
    error = expect - u.array_r
    mx, = fine_grid.getSizes()
    Hx = 1/(mx - 1)
    return Hx*numpy.linalg.norm(error)


def test_mms_convergence(coarse_grid):
    errors = []
    for nref in [1, 2, 3, 4, 5]:
        hierarchy = GridHierarchy(coarse_grid, nrefinements=nref)
        error = run_solve(hierarchy)
        errors.append(error)
    errors = numpy.asarray(errors)
    assert (numpy.log2(errors[:-1] / errors[1:]) > 2).all()


@pytest.mark.parametrize("lu", [False, True],
                         ids=["Jacobi coarse grid", "Exact coarse grid"])
def test_two_grid(coarse_grid, lu):
    hierarchy = GridHierarchy(coarse_grid, nrefinements=1) 
    operator = Poisson3pt

    fine = hierarchy[-1]
    u = fine.createGlobalVector()
    b, expect = rhs(fine)

    solver = MGSolver(hierarchy, operator)
    Af = solver.get_operator(1)
    Ac = solver.get_operator(0)

    P = solver.prolongations[0]

    for _ in range(10):
        r = solver.residuals[1]
        b_coarse = solver.rhses[0]
        u_coarse = solver.corrections[0]
        r_coarse = solver.residuals[0]
        # presmooth
        solver.jacobi(Af, u, b, r, alpha=0.8, niter=1)
        Af.mult(u, r)           # r <- Af u
        r.aypx(-1, b)           # r <- b - r
        P.multTranspose(r, b_coarse)  # b_coarse <- P^T r
        u_coarse.set(0)
        if lu:
            solver.lu_solve(Ac, u_coarse, b_coarse)
        else:
            solver.jacobi(Ac, u_coarse, b_coarse, r_coarse,
                          alpha=1, niter=15)
        P.multAdd(u_coarse, u, u)  # u <- u + P u_coarse
        # postsmooth
        solver.jacobi(Af, u, b, r, alpha=0.8, niter=1)
    assert numpy.allclose(u.array_r - expect, 0, atol=2e-4)
