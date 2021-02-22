from enum import Enum, auto
from functools import cached_property

from .operator import AbstractOperator
from .petsc import PETSc


class CycleType(Enum):
    """The multigrid cycle type (V- or W-cycle)."""
    V = auto()
    W = auto()


class MGSolver:
    Type = CycleType
    """Alias for CycleType"""

    def __init__(self, hierarchy, operator, galerkin=False):
        """Create a multigrid solver with the given hierarchy.
        :arg hierarchy: A :class:`GridHierarchy`
        :arg operator: class used to construct operators (a subclass
            of :class:`AbstractOperator`).
        :arg galerkin: Use galerkin coarse grids? If False, use
            rediscretisation.
        """
        self.hierarchy = hierarchy
        """The grid hierarchy"""
        self.operator = operator
        """Class to construct the operator on a given grid level"""
        self.comm = hierarchy.comm
        """MPI Communicator"""
        self.galerkin = galerkin
        """Should we use galerkin coarse grids?"""
        self.operators = {}
        """Cache for operators on levels."""
        self.coarse_grid_solver = None
        """Direct solver on the coarse grid."""

    @cached_property
    def residuals(self):
        """Storage for residuals on each level."""
        return tuple(grid.createGlobalVector() for grid in self.hierarchy)

    @cached_property
    def corrections(self):
        """Storage for corrections on each level."""
        return tuple(grid.createGlobalVector() for grid in self.hierarchy)

    @cached_property
    def rhses(self):
        """Storage of the right hand sides of each level."""
        return tuple(grid.createGlobalVector() for grid in self.hierarchy)

    @cached_property
    def prolongations(self):
        """Prolongators from level i to level i+1."""
        return tuple(c.createInterpolation(f)[0]
                     for c, f in zip(self.hierarchy, self.hierarchy[1:]))

    @cached_property
    def injections(self):
        """Injection from level i to level i+1."""
        return tuple(c.createInjection(f)
                     for c, f in zip(self.hierarchy, self.hierarchy[1:]))

    def get_operator(self, level):
        """Get an operator for a specified level.

        :arg level: The level in the hierarchy to get the operator.
        :returns: An operator (possibly from a cache) on the given level.
        """
        try:
            return self.operators[level]
        except KeyError:
            if self.galerkin:
                if level == len(self.hierarchy) - 1:
                    A = self.operator(self.hierarchy[level])
                    if not isinstance(A, AbstractOperator):
                        raise ValueError("Expecting an operator class that "
                                         "constructs instances of AbstractOperator")
                    A = A.as_sparse_matrix()
                else:
                    Af = self.get_operator(level+1)
                    P = self.prolongations[level]
                    A = Af.PtAP(P, fill=1)
            else:
                A = self.operator(self.hierarchy[level])
                if not isinstance(A, AbstractOperator):
                    raise ValueError("Expecting an operator class that "
                                     "constructs instances of AbstractOperator")
            return self.operators.setdefault(level, A)

    def solve(self, x, b, rtol=1e-8, maxiter=100, monitor=True,
              alpha=1, presmooth=1, postsmooth=1, cycle_type=CycleType.V):
        """Solve A x = b with the operator defined by this MGSolver.

        :arg x: The Vec in which to put the solution.
        :arg b: The right hand side.
        :arg rtol: Relative residual reduction after which to terminate.
        :arg maxiter: Maximum number of iterations to run.
        :arg monitor: Print monitoring information about the solve?
        :arg alpha: Scaling for the jacobi iteration smoothing steps.
        :arg presmooth: Number of presmoothing steps.
        :arg postsmooth: Number of postsmoothing steps.
        :arg cycle_type: The cycle type (CycleType.V or CycleType.W)."""
        if cycle_type == CycleType.V:
            cycle = self.vcycle
        elif cycle_type == CycleType.W:
            cycle = self.wcycle
            raise ValueError(f"Not a valid cycle type {cycle_type})")
        with PETSc.Log.Event("MGSolve"):
            nlevel = len(self.hierarchy)
            A = self.get_operator(nlevel - 1)
            r = self.residuals[nlevel - 1]
            A.mult(x, r)    # r <- Ax
            r.aypx(-1, b)   # r <- -1*r + b
            r0norm = r.norm()   # r0norm <- ||r||
            rnorm = r0norm      # before we start they're the same.
            if monitor:
                PETSc.Sys.Print(f"Iteration 0; rnorm {rnorm/r0norm:.4e} {rnorm:.4e}",
                                comm=self.comm)
            for i in range(1, maxiter+1):
                if rnorm/r0norm < rtol:
                    break
                cycle(nlevel - 1, x, b, alpha=alpha,
                      presmooth=presmooth, postsmooth=postsmooth)
                A.mult(x, r)    # r <- Ax
                r.aypx(-1, b)   # r <- -1*r + b
                rnorm = r.norm()
                if monitor:
                    PETSc.Sys.Print(f"Iteration {i}; rnorm {rnorm/r0norm:.4e} {rnorm:.4e}",
                                    comm=self.comm)

    def lu_solve(self, A, x, b):
        """Do an exact solve (via LU decomposition)
        :arg A: the operator.
        :arg x: the solution vector.
        :arg b: the right hand side.
        """
        if self.coarse_grid_solver is None:
            if isinstance(A, AbstractOperator):
                A = A.as_sparse_matrix()
            elif not isinstance(A, PETSc.Mat):
                raise ValueError("Expecting an AbstractOperator or PETSc.Mat"
                                 f" not a {type(A)}")
            pc = PETSc.PC().create(comm=A.comm)
            optdb = PETSc.Options()
            optdb["coarse_grid_solve_pc_type"] = "redundant"
            optdb["coarse_grid_solve_pc_redundant_type"] = "lu"
            pc.setOptionsPrefix("coarse_grid_solve_")
            pc.setOperators(A)
            pc.setFromOptions()
            pc.setUp()
            self.coarse_grid_solver = pc
        self.coarse_grid_solver.apply(b, x)

    def jacobi(self, A, x, b, r, alpha=1, niter=1):
        """
        Do a specified number of Jacobi iterations.

        x_{i+1} = x_i + alpha D^{-1}(b - Ax_i)

        :arg A: the operator.
        :arg x: the solution vector to update.
        :arg b: the right hand side.
        :arg r: storage to put the residual in.
        :arg alpha: scaling to apply to the update.
        :arg niter: The number of iterations.
        """
        raise NotImplementedError("Please implement me!")

    def vcycle(self, level, correction, rhs, alpha=1,
               presmooth=1, postsmooth=1):
        """Perform a V-cycle starting at the specified level.

        :arg level: The level to start at.
        :arg correction: The vector in which to place the correction.
        :arg rhs: The right hand side of the error equation.
        :arg alpha: scaling to apply to the jacobi smoothing steps.
        :arg presmooth: Number of presmoothing steps.
        :arg postsmooth: Number of postsmoothing steps.
        """
        raise NotImplementedError("Please implement me!")

    def wcycle(self, level, correction, rhs, alpha=1,
               presmooth=1, postsmooth=1):
        """Perform a W-cycle starting at the specified level.

        :arg level: The level to start at.
        :arg correction: The vector in which to place the correction.
        :arg rhs: The right hand side of the error equation.
        :arg alpha: scaling to apply to the jacobi smoothing steps.
        :arg presmooth: Number of presmoothing steps.
        :arg postsmooth: Number of postsmoothing steps.
        """
        raise NotImplementedError("Please implement me!")
