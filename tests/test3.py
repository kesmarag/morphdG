import numpy as np
import morphdg as mdg

mesh = mdg.Mesh()
mesh.load("mesh.dat")   
mesh.agglomerate(2048, 42)
# mesh.plot()

solver = mdg.DGSolver(mesh)

p_array = 3 * np.ones(mesh.num_elements, dtype=np.int32)
# p_array[1] = 3 
solver.update_p_orders(p_array)

solver.set_params(
    vx = "10.0* Kokkos::sin(x) + 0.2 * y",
    vy = "1.0",
    Kxx = "1.0",
    Kyy = "0.5",
    alpha = 5.0
)

solver.set_source(0.0)
solver.set_dirichlet_bc(loc=lambda x, y: x < 0.01, dirichlet_input="100* Kokkos::sin(y)")
solver.set_neumann_bc(loc=lambda x, y: x > 0.99, neumann_input="10.0")
solver.set_dirichlet_bc(loc=lambda x, y: y > 0.99, dirichlet_input="0.0")
solver.set_dirichlet_bc(loc=lambda x, y: y < 0.01, dirichlet_input="0.0")

solution = solver.solve(mode="kokkos")

# print(solution)

solver.plot_solution(solution, "dg_solution.png")

del solver, mesh
