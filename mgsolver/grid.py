from .petsc import PETSc


class AbstractGrid:
    pass


class Grid1D(AbstractGrid):
    def __init__(self, nx, stencil_width=1, comm=PETSc.COMM_WORLD):
        """Create a 1D grid.

        :arg nx: Number of vertices in x direction.
        :arg stencil_width: Width of stencil (1 for 3-point)
        :arg comm: Communicator to build the grid on.
        """
        self.grid = PETSc.DMDA().create(dof=1, sizes=(nx, ),
                                        stencil_width=stencil_width,
                                        stencil_type=PETSc.DMDA.StencilType.STAR,
                                        comm=comm,
                                        setup=False)
        self.grid.setFromOptions()


class Grid2D(AbstractGrid):
    def __init__(self, nx, ny, stencil_width=1, comm=PETSc.COMM_WORLD):
        """Create a 2D grid.

        :arg nx: Number of vertices in x direction.
        :arg ny: Number of vertices in y direction.
        :arg stencil_width: Width of stencil (1 for 5-point)
        :arg comm: Communicator to build the grid on.
        """
        self.grid = PETSc.DMDA().create(dof=1, sizes=(nx, ny),
                                        stencil_width=stencil_width,
                                        stencil_type=PETSc.DMDA.StencilType.STAR,
                                        comm=comm,
                                        setup=False)
        self.grid.setFromOptions()


class Grid3D(AbstractGrid):
    def __init__(self, nx, ny, nz, stencil_width=1, comm=PETSc.COMM_WORLD):
        """Create a 3D grid.

        :arg nx: Number of vertices in x direction.
        :arg ny: Number of vertices in y direction.
        :arg nz: Number of vertices in z direction.
        :arg stencil_width: Width of stencil (1 for 7-point)
        :arg comm: Communicator to build the grid on.
        """
        self.grid = PETSc.DMDA().create(dof=1, sizes=(nx, ny, nz),
                                        stencil_width=stencil_width,
                                        stencil_type=PETSc.DMDA.StencilType.STAR,
                                        comm=comm,
                                        setup=False)
        self.grid.setFromOptions()


class GridHierarchy:
    def __init__(self, grid, nrefinements=1):
        """Build a hierarchy of grids.
        :arg grid: Coarse grid to start from.
        :arg nrefinements: Number of refinements."""
        if not isinstance(grid, AbstractGrid):
            raise ValueError("Expecting a Grid2D or Grid3D, not a "
                             f"{type(grid)}")
        grid = grid.grid
        grid.setUp()
        self.comm = grid.comm
        hierarchy = grid.refineHierarchy(nrefinements)
        self.hierarchy = (grid, *hierarchy)
        for grid in self.hierarchy:
            grid.setUp()
            grid.setUniformCoordinates()

    def __getitem__(self, idx):
        return self.hierarchy[idx]

    def __iter__(self):
        yield from iter(self.hierarchy)

    def __len__(self):
        return len(self.hierarchy)
