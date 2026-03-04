import numpy as np
import morphdg as mdg

mesh = mdg.Mesh()
mesh.load("mesh.dat")   
mesh.agglomerate(1024, 42)
# mesh.plot()

solver = mdg.DGSolver(mesh, p_order=1, alpha=5.0)

solver.set_pde_params(
    vx = 0.0, 
    vy = 1.0,
    Kxx = lambda x, y: x+y, 
    Kyy = 1.0
)

solver.set_source(0.0)

solver.set_dirichlet_bc(loc=lambda x, y: x < 0.01, dirichlet_input=100.0)
solver.set_neumann_bc(loc=lambda x, y: x > 0.99, neumann_input=0.0)
solver.set_neumann_bc(loc=lambda x, y: y > 0.99, neumann_input=0.0)
solver.set_neumann_bc(loc=lambda x, y: y < 0.01, neumann_input=0.0)

solution = solver.solve(mode="kokkos")
# solver.plot_solution(solution, "dg_solution.png")

del solver, mesh
