from setuptools import find_packages, setup

setup(name="mgsolver",
      version="0.1",
      description="A simple multigrid solver using PETSc",
      install_requires=["numpy",
                        "numba",
                        "Cython",
                        "flake8",
                        "pytest",
                        "petsc4py"],
      packages=find_packages())
