import abc


class AbstractOperator(metaclass=abc.ABCMeta):
    def __init__(self, grid):
        """An operator on a particular grid
        :arg grid: the grid (a level from a :class:`GridHierarchy`)"""
        self.grid = grid

    def getDiagonal(self):
        """Return the diagonal of the operator as a PETSc Vec."""
        return self.diagonal

    @abc.abstractproperty
    def diagonal(self):
        """Return the diagonal of the operator as a PETSc Vec.

        Create the Vec with self.grid.createGlobalVector()
        """

    @abc.abstractmethod
    def mult(self, x, y):
        """Perform matrix-vector multiplication y = Ax.

        :arg x: The vector to multiply (a PETSc Vec).
        :arg y: The vector in which to place the result (a PETSc Vec).
        """

    @abc.abstractmethod
    def as_sparse_matrix(self):
        """Return the operator as a sparse matrix.

        Use self.grid.createMatrix() to get the storage for the matrix.
        """
