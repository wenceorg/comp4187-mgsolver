import sys

import petsc4py

petsc4py.init(sys.argv)

from petsc4py import PETSc  # noqa: E402

PETSc.Sys.popErrorHandler()
