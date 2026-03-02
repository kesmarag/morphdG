import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Import the Pybind11 C++ module
import morphdg as mdg

# ==============================================================================
# 1. INITIALIZE KOKKOS & MESH
# ==============================================================================
print("1. Starting Kokkos Engine...")
engine = mdg.KokkosManager()

print("2. Loading and Agglomerating Mesh...")
mesh = mdg.AggMesh()
mesh.load("mesh.dat")            # Ensure mesh.dat is in the same directory!
mesh.random_agglomerate(1024,42)  # Create 16 random polygonal elements
mesh.push_to_device()            # Deep copy geometry to the GPU
mesh.plot()
# ==============================================================================
# 2. SETUP SOLVER & POLYNOMIALS
# ==============================================================================
print("3. Initializing DG Solver...")
solver = mdg.DGSolver()
solver.coeffs.alpha = 5.0  # SIPG penalty parameter

# Set all elements to use Linear basis functions (P=1)
p_orders = np.full(mesh.num_elements, 1, dtype=np.int32)
solver.set_p_orders(p_orders)

# ==============================================================================
# 3. VOLUME PHYSICS (The Interior Domain)
# ==============================================================================
print("4. Evaluating Volume Physics...")
X_vol, Y_vol = solver.vol_quad_points(mesh)
X_vol, Y_vol = np.array(X_vol), np.array(Y_vol)

# Set diagonal advection velocity (vx=1.0, vy=1.0)
solver.set_vx_field(5.0*np.ones_like(X_vol))
solver.set_vy_field(np.zeros_like(X_vol))

# Set isotropic diffusion (Kxx=1.0, Kyy=1.0)
solver.set_Kxx_field(np.ones_like(X_vol))
solver.set_Kyy_field(np.ones_like(X_vol))
solver.set_Kxy_field(np.zeros_like(X_vol))
solver.set_Kyx_field(np.zeros_like(X_vol))

# Apply a sine-wave heat source driving the PDE
source_term = 1.0*np.sin(np.pi * X_vol) * np.cos(np.pi * Y_vol)
solver.set_source_nodal(source_term)

# ==============================================================================
# 4. FACE PHYSICS (Velocity and Diffusion along edges)
# ==============================================================================
print("5. Evaluating Face Physics...")
X_face, Y_face = solver.face_quad_points(mesh)
X_face, Y_face = np.array(X_face), np.array(Y_face)

# Edges share the exact same physical properties as the volume
solver.set_vx_face(5.0*np.ones_like(X_face))
solver.set_vy_face(np.zeros_like(X_face))
solver.set_Kxx_face(np.ones_like(X_face))
solver.set_Kyy_face(np.ones_like(X_face))
solver.set_Kxy_face(np.zeros_like(X_face))
solver.set_Kyx_face(np.zeros_like(X_face))

# ==============================================================================
# 5. MIXED BOUNDARY CONDITIONS (Redundancy-Free Architecture)
# ==============================================================================
print("6. Applying Mixed Boundary Conditions...")

# 5a. Find the exact split point between internal and boundary faces
# mesh.faces is a 1D flat array [ePlus, eMinus, nA, nB, ...], so divide by 4
num_int_faces = len(mesh.faces) // 4
# 2 quadrature points per face
split_idx = num_int_faces * 2 

# 5b. Extract ONLY the boundary coordinates!
X_bnd = X_face[split_idx:]
Y_bnd = Y_face[split_idx:]

# 5c. Allocate boundary condition arrays sized perfectly to the boundary
g_D_bnd = np.zeros_like(X_bnd)    # Dirichlet values (u = g)
g_N_bnd = np.zeros_like(X_bnd)    # Neumann fluxes (grad u * n = h)
bctype_bnd = np.zeros_like(X_bnd) # 0 = Dirichlet, 1 = Neumann

# Select ONLY the actual left and right walls
left_wall_mask = X_bnd < 0.01
right_wall_mask = X_bnd > 0.99

# Left Wall: Hot Radiator (Fixed at 100.0)
bctype_bnd[left_wall_mask] = 0.0
g_D_bnd[left_wall_mask] = 100.0

# Right Wall: Open Window (Neumann Flux = 0.0 allows heat to exit naturally)
bctype_bnd[right_wall_mask] = 1.0 
g_N_bnd[right_wall_mask] = 0.0  

# (Top and Bottom walls remain the default Dirichlet value u = 0.0)

# 5e. Push these perfectly sized arrays directly to C++
solver.set_g_D_face(g_D_bnd)
solver.set_g_N_face(g_N_bnd)
solver.set_bctype_face(bctype_bnd)

# ==============================================================================
# 6. ASSEMBLE SYSTEM & EXTRACT MATRIX
# ==============================================================================
print("7. Assembling Global Matrix on GPU...")
solver.create_sparse_graph(mesh)
solver.assemble(mesh)

solver.print_matrix(15)

print("8. Extracting CSR Matrix to Python...")
vals, cols, rows, rhs = solver.get_global_system()

print(f"   -> Matrix size: {len(rhs)}x{len(rhs)} DOFs")
print(f"   -> Non-zeros:   {len(vals)}")

# ==============================================================================
# 7. SOLVE WITH SCIPY
# ==============================================================================
# Build the SciPy Sparse CSR Matrix
A = sp.csr_matrix((vals, cols, rows), shape=(len(rhs), len(rhs)))

print("9. Solving with SciPy SuperLU Backend...")
solution = spla.spsolve(A, rhs)

# ==============================================================================
# 8. RESULTS
# ==============================================================================
print("\n=== SOLVE COMPLETE ===")
print(f"Solution Min Value:  {np.min(solution):.4f}")
print(f"Solution Max Value:  {np.max(solution):.4f}")
print(f"Solution Mean Value: {np.mean(solution):.4f}")


# ==============================================================================
# 9. VISUALIZATION (Reconstructing the DG Polynomials)
# ==============================================================================
import matplotlib.pyplot as plt

print("10. Reconstructing High-Order solution for plotting...")

# Pybind11 returns 1D flat arrays, so we reshape them to standard 2D formats
nodes = mesh.nodes.reshape(-1, 2) 
tris = mesh.triangles.reshape(-1, 3)
t_offsets = mesh.t_offsets
n_elem = mesh.num_elements

# We will evaluate the PDE solution at the centroid of every fine triangle
tri_vals = np.zeros(len(tris))

for e in range(n_elem):
    start_tri = t_offsets[e]
    end_tri   = t_offsets[e+1]
    
    # 1. Recompute the Bounding Box for this specific polygon
    elem_nodes = nodes[tris[start_tri:end_tri].flatten()]
    min_x, max_x = elem_nodes[:, 0].min(), elem_nodes[:, 0].max()
    min_y, max_y = elem_nodes[:, 1].min(), elem_nodes[:, 1].max()
    dx, dy = max_x - min_x, max_y - min_y
    
    scale_x = np.sqrt(2.0 / dx)
    scale_y = np.sqrt(2.0 / dy)
    
    # 2. Get the DG coefficients for this element (P=1 means 3 DOFs per element)
    dof_start = e * 3
    c0, c1, c2 = solution[dof_start : dof_start+3]
    
    # 3. Evaluate the exact polynomial for each fine triangle
    for t in range(start_tri, end_tri):
        n1, n2, n3 = tris[t]
        
        cx = (nodes[n1, 0] + nodes[n2, 0] + nodes[n3, 0]) / 3.0
        cy = (nodes[n1, 1] + nodes[n2, 1] + nodes[n3, 1]) / 3.0
        
        xi  = 2.0 * (cx - min_x) / dx - 1.0
        eta = 2.0 * (cy - min_y) / dy - 1.0
        
        v0 = scale_x * 0.70710678 * scale_y * 0.70710678
        v1 = scale_x * (1.22474487 * xi) * scale_y * 0.70710678
        v2 = scale_x * 0.70710678 * scale_y * (1.22474487 * eta)
        
        # This converts the coefficients back into REAL temperatures!
        tri_vals[t] = c0*v0 + c1*v1 + c2*v2

print("11. Rendering Image...")
plt.figure(figsize=(10, 8))

# Plot the discontinuous field using 'tripcolor' with flat shading
plt.tripcolor(nodes[:, 0], nodes[:, 1], tris, facecolors=tri_vals, 
              cmap='inferno', edgecolors='k', linewidth=0.1)

plt.colorbar(label='Physical Field Value (Temperature/Velocity)')
plt.title('Discontinuous Galerkin Solution')
plt.axis('equal')
plt.tight_layout()
plt.savefig('dg_solution.png', dpi=300)
print(" -> Saved to 'dg_solution.png'!")

# ==============================================================================
# CLEANUP (Must destroy in order to satisfy Pybind11 references!)
# ==============================================================================
# 1. Delete the NumPy arrays holding onto the mesh memory
del nodes
del tris
del t_offsets
del tri_vals

# 2. Delete the Solver and Mesh (This triggers C++ memory deallocation)
del solver
del mesh

# 3. Delete the Engine (This safely calls Kokkos::finalize() last!)
del engine
