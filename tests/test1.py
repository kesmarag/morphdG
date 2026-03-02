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
mesh.random_agglomerate(16,42)  # Create 16 random polygonal elements
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
solver.set_vx_field(np.ones_like(X_vol))
solver.set_vy_field(np.ones_like(X_vol))

# Set isotropic diffusion (Kxx=1.0, Kyy=1.0)
solver.set_Kxx_field(np.ones_like(X_vol))
solver.set_Kyy_field(np.ones_like(X_vol))
solver.set_Kxy_field(np.zeros_like(X_vol))
solver.set_Kyx_field(np.zeros_like(X_vol))

# Apply a sine-wave heat source driving the PDE
source_term = 0.0*np.sin(np.pi * X_vol) * np.cos(np.pi * Y_vol) + 10.0
solver.set_source_nodal(source_term)

# ==============================================================================
# 4. FACE PHYSICS (Velocity and Diffusion along edges)
# ==============================================================================
print("5. Evaluating Face Physics...")
X_face, Y_face = solver.face_quad_points(mesh)
X_face, Y_face = np.array(X_face), np.array(Y_face)

# Edges share the exact same physical properties as the volume
solver.set_vx_face(np.ones_like(X_face))
solver.set_vy_face(np.ones_like(X_face))
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

# 5d. Define physical boundary regions 
left_wall_mask = X_bnd < 0.01
right_wall_mask = X_bnd > 0.99

# Left Wall: Dirichlet condition (Fixed value u = 10.0)
bctype_bnd[left_wall_mask] = 0.0
g_D_bnd[left_wall_mask] = 100.0

# Right Wall: Neumann condition (Heat/Fluid flux of 5.0 leaving domain)
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

# Cleanup to ensure Kokkos shuts down gracefully
del solver
del mesh
del engine
