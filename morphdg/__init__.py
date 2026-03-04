from . import morphdg_core

KokkosManager = morphdg_core.KokkosManager()

# AggMesh = morphdg_core.AggMesh

# DGSolver = morphdg_core.DGSolver
# Coeffs = morphdg_core.Coeffs

# from .plot_mesh import _plot_mesh
# AggMesh.plot = _plot_mesh

from .mesh import Mesh

from .dg_solver import DGSolver
