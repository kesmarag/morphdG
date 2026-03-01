import morphdg
import numpy as np


print("Setting up Kokkos Engine...")
engine = morphdg.KokkosManager()

print("Initializing AggMesh...")
mesh = morphdg.AggMesh()

mesh.load("./mesh.dat") 
mesh.random_agglomerate(2, 42)

mesh.push_to_device()

# print("Plotting...")
mesh.plot()

solver = morphdg.DGSolver()

# solver.coeffs.alpha = 0.0

x_list, y_list = solver.vol_quad_points(mesh)

n_elems = mesh.num_elements

p_orders = np.full(n_elems, 1, dtype=np.int32)
solver.set_p_orders(p_orders)



x = np.array(x_list)
y = np.array(y_list)

vx_field =  0.0 * y
vy_field =  0.0 * x

# 3. Pass the zero-copy arrays straight into Kokkos
solver.set_vx_field(vx_field)
solver.set_vy_field(vy_field)

print(f"Loaded a velocity field with {len(vx_field)} quadrature points.")


# --- ADD THIS: THE MISSING SOURCE TERM ---
source_field = np.sin(np.pi * x) * np.cos(np.pi * y) # Example source function
solver.set_source_nodal(source_field)


# Isotropic standard diffusion (K = 1.0)
Kxx_field = np.ones_like(x)
Kyy_field = np.ones_like(x)

# Zero off-diagonal terms
Kxy_field = np.zeros_like(x)
Kyx_field = np.zeros_like(x)

solver.set_Kxx_field(Kxx_field)
solver.set_Kyy_field(Kyy_field)
solver.set_Kxy_field(Kxy_field)
solver.set_Kyx_field(Kyx_field)


solver.create_sparse_graph(mesh)

print("4. Launching GPU Volume Kernel...")

# Step C: Assemble!
solver.assemble_volume(mesh)


solver.print_matrix(15)

del solver
del mesh
del engine
