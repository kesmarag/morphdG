import numpy as np
import morphdg as mdg

mesh = mdg.Mesh()
mesh.load("mesh.dat")   
mesh.agglomerate(1024, 42)
mesh.plot()

solver = mdg.DGSolver(mesh)

p_array = np.ones(mesh.num_elements, dtype=np.int32)
# p_array[2] = 2

solver.update_p_orders(p_array)

solver.set_params(
    vx = lambda x, y: 0.1*x+0.2*y, 
    vy = 1.0,
    Kxx = lambda x, y: 10*x+y, 
    Kyy = 1.0,
    alpha = 5.0
)

solver.set_source(0.0)

solver.set_dirichlet_bc(loc=lambda x, y: x < 0.01, dirichlet_input=100.0)
solver.set_neumann_bc(loc=lambda x, y: x > 0.99, neumann_input=0.0)
solver.set_neumann_bc(loc=lambda x, y: y > 0.99, neumann_input=1.0)
solver.set_neumann_bc(loc=lambda x, y: y < 0.01, neumann_input=0.0)

solution = solver.solve(mode="kokkos")
solver.plot_solution(solution, "dg_solution.png")

del solver, mesh
