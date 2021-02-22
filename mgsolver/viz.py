import os
from itertools import chain

from .petsc import PETSc


def write_output(basename, vector, *vectors):
    """Write the provided vectors to output for visualisation in paraview.

    :arg basename: VTK .vtr file to write to (if no extension given,
        will append .vtr).
    :arg vector: First vector to write.
    :arg vectors: Remaining vectors to write. Give them meaningful
        names by saying vec.setName("foo").

    .. warning::

       Overwrites the provided file if it already exists.
    """
    basename, ext = os.path.splitext(basename)
    if ext in {"vtr", ""}:
        filename = os.path.abspath(f"{basename}.vtr")
    else:
        filename = os.path.abspath(f"{basename}.{ext}.vtr")
    viewer = PETSc.Viewer().create(comm=vector.comm)
    viewer.setType(viewer.Type.VTK)
    viewer.setFileName(filename)
    viewer.setFileMode(viewer.Mode.WRITE)
    viewer.pushFormat(viewer.Format.VTK_VTR)
    for vec in chain([vector], vectors):
        vec.view(viewer)
    viewer.popFormat()
    viewer.destroy()
