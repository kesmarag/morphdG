import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

from . import morphdg_core
from .jit import JITEvaluator  # Import our new JIT compiler

class DGSolver:

    def __init__(self, mesh, p_order=1):
        # mesh is the Python AggMesh object
        self.mesh = mesh 
        self._cpp_solver = morphdg_core.DGSolver()
        
        # Set polynomial order and FIX the missing class attribute
        self.p_orders = np.full(self.mesh.num_elements, p_order, dtype=np.int32)
        self._cpp_solver.set_p_orders(self.p_orders)
        
        # Internal boundary tracking
        self._dirichlet_conditions = []
        self._neumann_conditions = []

    def _eval_any(self, field_input, X, Y):
        """
        Evaluates input fields. 
        If it's a string, it triggers the C++ Kokkos JIT compiler for maximum performance.
        Otherwise, it falls back to standard NumPy/Python evaluation.
        """
        # --- THE JIT COMPILER APPROACH ---
        if isinstance(field_input, str):
            # Create a blank array to hold the Kokkos output
            out_array = np.zeros_like(X, dtype=np.float64)
            
            # Ensure X and Y are contiguous in memory (required for C++ pointers)
            X_contig = np.ascontiguousarray(X, dtype=np.float64)
            Y_contig = np.ascontiguousarray(Y, dtype=np.float64)
            
            # Compile the C++ string (or load it from cache instantly)
            jit_eval = JITEvaluator(field_input)
            
            # Get the raw memory addresses from the NumPy arrays
            ptr_out = out_array.ctypes.data
            ptr_x = X_contig.ctypes.data
            ptr_y = Y_contig.ctypes.data
            n_points = len(X)
            
            # Execute the compiled Kokkos kernel! (Time t=0.0 for steady-state setup)
            jit_eval.evaluate(ptr_out, ptr_x, ptr_y, n_points, 0.0)
            
            return out_array
            
        # --- THE STANDARD PYTHON FALLBACK APPROACH ---
        elif isinstance(field_input, np.ndarray):
            val = field_input
        elif callable(field_input):
            try:
                val = field_input(X, Y)
            except Exception:
                val = np.vectorize(field_input)(X, Y)
        else:
            val = field_input
            
        # .copy() ensures the memory is contiguous so Pybind11/C++ doesn't panic.
        return np.broadcast_to(val, X.shape).astype(np.float64).copy()
        
    def set_params(self, vx=0.0, vy=0.0, Kxx=1.0, Kyy=1.0, Kxy=0.0, Kyx=0.0, alpha=5.0):
        # Volume points
        X_vol, Y_vol = self._cpp_solver.vol_quad_points(self.mesh._cpp_mesh)
        X_vol, Y_vol = np.array(X_vol), np.array(Y_vol)

        self._cpp_solver.set_vx_field(self._eval_any(vx, X_vol, Y_vol))
        self._cpp_solver.set_vy_field(self._eval_any(vy, X_vol, Y_vol))
        self._cpp_solver.set_Kxx_field(self._eval_any(Kxx, X_vol, Y_vol))
        self._cpp_solver.set_Kyy_field(self._eval_any(Kyy, X_vol, Y_vol))
        self._cpp_solver.set_Kxy_field(self._eval_any(Kxy, X_vol, Y_vol))
        self._cpp_solver.set_Kyx_field(self._eval_any(Kyx, X_vol, Y_vol))

        # Face points
        X_face, Y_face = self._cpp_solver.face_quad_points(self.mesh._cpp_mesh)
        X_face, Y_face = np.array(X_face), np.array(Y_face)

        self._cpp_solver.set_vx_face(self._eval_any(vx, X_face, Y_face))
        self._cpp_solver.set_vy_face(self._eval_any(vy, X_face, Y_face))
        self._cpp_solver.set_Kxx_face(self._eval_any(Kxx, X_face, Y_face))
        self._cpp_solver.set_Kyy_face(self._eval_any(Kyy, X_face, Y_face))
        self._cpp_solver.set_Kxy_face(self._eval_any(Kxy, X_face, Y_face))
        self._cpp_solver.set_Kyx_face(self._eval_any(Kyx, X_face, Y_face))

        self._cpp_solver.coeffs.alpha = alpha

    def set_dirichlet_bc(self, loc, dirichlet_input):
        self._dirichlet_conditions.append((loc, dirichlet_input))

    def set_neumann_bc(self, loc, neumann_input):
        self._neumann_conditions.append((loc, neumann_input))

    def set_source(self, source_input):
        X_vol, Y_vol = self._cpp_solver.vol_quad_points(self.mesh._cpp_mesh)
        X_vol, Y_vol = np.array(X_vol), np.array(Y_vol)
        self._cpp_solver.set_source_nodal(self._eval_any(source_input, X_vol, Y_vol))


    # def _apply_boundaries(self):
    #     X_face, Y_face = self._cpp_solver.face_quad_points(self.mesh._cpp_mesh)
    #     X_face, Y_face = np.array(X_face), np.array(Y_face)
        
    #     num_int_faces = len(self.mesh._cpp_mesh.faces) // 4
    #     split_idx = num_int_faces * 2 
        
    #     X_bnd = X_face[split_idx:]
    #     Y_bnd = Y_face[split_idx:]
        
    #     g_D = np.zeros_like(X_bnd)
    #     g_N = np.zeros_like(X_bnd)
    #     bctype = np.zeros_like(X_bnd) # Default to 0=Dirichlet
        
    #     for locator, value in self._dirichlet_conditions:
    #         mask = locator(X_bnd, Y_bnd)
    #         g_D[mask] = self._eval_any(value, X_bnd[mask], Y_bnd[mask])
    #         bctype[mask] = 0.0
            
    #     for locator, value in self._neumann_conditions:
    #         mask = locator(X_bnd, Y_bnd)
    #         g_N[mask] = self._eval_any(value, X_bnd[mask], Y_bnd[mask])
    #         bctype[mask] = 1.0

    #     self._cpp_solver.set_g_D_face(g_D)
    #     self._cpp_solver.set_g_N_face(g_N)
    #     self._cpp_solver.set_bctype_face(bctype)

    def _apply_boundaries(self):
        X_face, Y_face = self._cpp_solver.face_quad_points(self.mesh._cpp_mesh)
        X_face, Y_face = np.array(X_face), np.array(Y_face)
        
        # Dynamically determine the number of face quadrature points!
        max_p = np.max(self.p_orders)
        # print(max_p)
        if max_p == 1: n_q_face = 2
        elif max_p == 2: n_q_face = 3
        elif max_p == 3: n_q_face = 4
        else: n_q_face = 2
        
        num_int_faces = len(self.mesh._cpp_mesh.faces) // 4
        split_idx = num_int_faces * n_q_face # Use dynamic points!
        
        X_bnd = X_face[split_idx:]
        Y_bnd = Y_face[split_idx:]
        
        g_D = np.zeros_like(X_bnd)
        g_N = np.zeros_like(X_bnd)
        bctype = np.zeros_like(X_bnd) # Default to 0=Dirichlet
        
        for locator, value in self._dirichlet_conditions:
            mask = locator(X_bnd, Y_bnd)
            g_D[mask] = self._eval_any(value, X_bnd[mask], Y_bnd[mask])
            bctype[mask] = 0.0
            
        for locator, value in self._neumann_conditions:
            mask = locator(X_bnd, Y_bnd)
            g_N[mask] = self._eval_any(value, X_bnd[mask], Y_bnd[mask])
            bctype[mask] = 1.0

        self._cpp_solver.set_g_D_face(g_D)
        self._cpp_solver.set_g_N_face(g_N)
        self._cpp_solver.set_bctype_face(bctype)

        
    def solve(self, mode="scipy", max_iters=10000, tolerance=1e-8):
        
        self._apply_boundaries()
        self._cpp_solver.create_sparse_graph(self.mesh._cpp_mesh)
        self._cpp_solver.assemble(self.mesh._cpp_mesh)
        
        if mode == "scipy":
            vals, cols, rows, rhs = self._cpp_solver.get_global_system()
            A = sp.csr_matrix((vals, cols, rows), shape=(len(rhs), len(rhs)))
            print("Solving with SciPy SuperLU Backend...")
            return spla.spsolve(A, rhs)
            
        elif mode == "kokkos_cg":
            print("Solving with Kokkos CG Backend (Symmetric)...")
            return self._cpp_solver.solve_cg(max_iters=max_iters, tolerance=tolerance)
            
        elif mode == "kokkos_bicgstab" or mode == "kokkos":
            print("Solving with Kokkos BiCGStab GPU (Non-Symmetric)...")
            return self._cpp_solver.solve_bicgstab(max_iters=max_iters, tolerance=tolerance)
            
        else:
            raise ValueError(f"Unknown backend '{mode}'.")

    def update_p_orders(self, p_order):
        if isinstance(p_order, int):
            self.p_orders = np.full(self.mesh.num_elements, p_order, dtype=np.int32)
        else:
            self.p_orders = np.asarray(p_order, dtype=np.int32)
        self._cpp_solver.set_p_orders(self.p_orders)
        print(f" -> p-orders updated! Matrix resized to {self._cpp_solver.total_dofs} DOFs.")    

    def assemble_system(self):
        """Assembles the matrices and boundary conditions for RK4 time-stepping."""
        self._apply_boundaries()
        self._cpp_solver.create_sparse_graph(self.mesh._cpp_mesh)
        self._cpp_solver.assemble(self.mesh._cpp_mesh)

    def advance_time(self, dt):
        """Steps the physics forward by dt using GPU RK4."""
        self._cpp_solver.advance_rk4(dt)
        return self._cpp_solver.get_state()


        
    # def plot_solution(self, solution, save_as="dg_solution.png"):
    #     import matplotlib.pyplot as plt
    #     from matplotlib.collections import LineCollection
    #     nodes = self.mesh._cpp_mesh.nodes.reshape(-1, 2) 
    #     tris = self.mesh._cpp_mesh.triangles.reshape(-1, 3)
    #     t_offsets = self.mesh._cpp_mesh.t_offsets
    #     n_elem = self.mesh.num_elements
        
    #     # 1. Create a "disconnected" mesh for plotting DG jumps perfectly
    #     n_plot_nodes = len(tris) * 3
    #     plot_x = np.zeros(n_plot_nodes)
    #     plot_y = np.zeros(n_plot_nodes)
    #     plot_v = np.zeros(n_plot_nodes)
    #     plot_tris = np.arange(n_plot_nodes).reshape(-1, 3)
        
    #     dof_offset = 0  
        
    #     for e in range(n_elem):
    #         p = self.p_orders[e]
    #         n_basis_e = (p + 1) * (p + 2) // 2
            
    #         coeffs = solution[dof_offset : dof_offset + n_basis_e]
    #         dof_offset += n_basis_e
            
    #         start_tri = t_offsets[e]
    #         end_tri   = t_offsets[e+1]
            
    #         elem_nodes = nodes[tris[start_tri:end_tri].flatten()]
    #         min_x, max_x = elem_nodes[:, 0].min(), elem_nodes[:, 0].max()
    #         min_y, max_y = elem_nodes[:, 1].min(), elem_nodes[:, 1].max()
    #         dx, dy = max_x - min_x, max_y - min_y
            
    #         scale_x, scale_y = np.sqrt(2.0 / dx), np.sqrt(2.0 / dy)
            
    #         for t in range(start_tri, end_tri):
    #             for k, n_idx in enumerate(tris[t]):
    #                 nx = nodes[n_idx, 0]
    #                 ny = nodes[n_idx, 1]
                    
    #                 xi  = 2.0 * (nx - min_x) / dx - 1.0
    #                 eta = 2.0 * (ny - min_y) / dy - 1.0
                    
    #                 L0_xi, L1_xi, L2_xi = 0.70710678, 1.22474487 * xi, 1.58113883 * (1.5 * xi**2 - 0.5)
    #                 L0_eta, L1_eta, L2_eta = 0.70710678, 1.22474487 * eta, 1.58113883 * (1.5 * eta**2 - 0.5)
                    
    #                 v0 = scale_x * L0_xi * scale_y * L0_eta
    #                 v1 = scale_x * L1_xi * scale_y * L0_eta
    #                 v2 = scale_x * L0_xi * scale_y * L1_eta
    #                 val = coeffs[0]*v0 + coeffs[1]*v1 + coeffs[2]*v2
                    
    #                 if n_basis_e >= 6:
    #                     v3 = scale_x * L2_xi * scale_y * L0_eta
    #                     v4 = scale_x * L1_xi * scale_y * L1_eta
    #                     v5 = scale_x * L0_xi * scale_y * L2_eta
    #                     val += coeffs[3]*v3 + coeffs[4]*v4 + coeffs[5]*v5

    #                 if n_basis_e >= 10:
    #                     L3_xi = 1.87082869 * (2.5 * xi**3 - 1.5 * xi)
    #                     L3_eta = 1.87082869 * (2.5 * eta**3 - 1.5 * eta)
                        
    #                     v6 = scale_x * L3_xi * scale_y * L0_eta
    #                     v7 = scale_x * L2_xi * scale_y * L1_eta
    #                     v8 = scale_x * L1_xi * scale_y * L2_eta
    #                     v9 = scale_x * L0_xi * scale_y * L3_eta
    #                     val += coeffs[6]*v6 + coeffs[7]*v7 + coeffs[8]*v8 + coeffs[9]*v9
                        
    #                 idx = t * 3 + k
    #                 plot_x[idx] = nx
    #                 plot_y[idx] = ny
    #                 plot_v[idx] = val

      
    #     fig, ax = plt.subplots(figsize=(10, 8))
        
   
    #     tc = ax.tripcolor(plot_x, plot_y, plot_tris, plot_v, 
    #                       shading='gouraud', cmap='inferno')
        
       
    #     faces = self.mesh._cpp_mesh.faces.reshape(-1, 4)
    #     bnd_faces = self.mesh._cpp_mesh.bnd_faces.reshape(-1, 3)
        
    #     segments = []
    #     for f in faces:
    #         nA, nB = f[2], f[3]
    #         segments.append([nodes[nA], nodes[nB]])
    #     for f in bnd_faces:
    #         nA, nB = f[1], f[2]
    #         segments.append([nodes[nA], nodes[nB]])
            
      
    #     lc = LineCollection(segments, colors='white', linewidths=0.25, zorder=3)
    #     ax.add_collection(lc)

    #     plt.colorbar(tc, label='Field Value')
    #     plt.title('morphdG : Discontinuous Galerkin Solution')
    #     plt.axis('equal')
    #     plt.tight_layout()
    #     plt.savefig(save_as, dpi=300)
    #     print(f" -> Saved to '{save_as}'!")
    #     plt.close()


    def plot_solution(self, solution, save_as="dg_solution.png", vlim=None):
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        nodes = self.mesh._cpp_mesh.nodes.reshape(-1, 2) 
        tris = self.mesh._cpp_mesh.triangles.reshape(-1, 3)
        t_offsets = self.mesh._cpp_mesh.t_offsets
        n_elem = self.mesh.num_elements
        
        # 1. Create a "disconnected" mesh for plotting DG jumps perfectly
        n_plot_nodes = len(tris) * 3
        plot_x = np.zeros(n_plot_nodes)
        plot_y = np.zeros(n_plot_nodes)
        plot_v = np.zeros(n_plot_nodes)
        plot_tris = np.arange(n_plot_nodes).reshape(-1, 3)
        
        dof_offset = 0  
        
        for e in range(n_elem):
            p = self.p_orders[e]
            n_basis_e = (p + 1) * (p + 2) // 2
            
            coeffs = solution[dof_offset : dof_offset + n_basis_e]
            dof_offset += n_basis_e
            
            start_tri = t_offsets[e]
            end_tri   = t_offsets[e+1]
            
            elem_nodes = nodes[tris[start_tri:end_tri].flatten()]
            min_x, max_x = elem_nodes[:, 0].min(), elem_nodes[:, 0].max()
            min_y, max_y = elem_nodes[:, 1].min(), elem_nodes[:, 1].max()
            dx, dy = max_x - min_x, max_y - min_y
            
            scale_x, scale_y = np.sqrt(2.0 / dx), np.sqrt(2.0 / dy)
            
            for t in range(start_tri, end_tri):
                for k, n_idx in enumerate(tris[t]):
                    nx = nodes[n_idx, 0]
                    ny = nodes[n_idx, 1]
                    
                    xi  = 2.0 * (nx - min_x) / dx - 1.0
                    eta = 2.0 * (ny - min_y) / dy - 1.0
                    
                    L0_xi, L1_xi, L2_xi = 0.70710678, 1.22474487 * xi, 1.58113883 * (1.5 * xi**2 - 0.5)
                    L0_eta, L1_eta, L2_eta = 0.70710678, 1.22474487 * eta, 1.58113883 * (1.5 * eta**2 - 0.5)
                    
                    v0 = scale_x * L0_xi * scale_y * L0_eta
                    v1 = scale_x * L1_xi * scale_y * L0_eta
                    v2 = scale_x * L0_xi * scale_y * L1_eta
                    val = coeffs[0]*v0 + coeffs[1]*v1 + coeffs[2]*v2
                    
                    if n_basis_e >= 6:
                        v3 = scale_x * L2_xi * scale_y * L0_eta
                        v4 = scale_x * L1_xi * scale_y * L1_eta
                        v5 = scale_x * L0_xi * scale_y * L2_eta
                        val += coeffs[3]*v3 + coeffs[4]*v4 + coeffs[5]*v5

                    if n_basis_e >= 10:
                        L3_xi = 1.87082869 * (2.5 * xi**3 - 1.5 * xi)
                        L3_eta = 1.87082869 * (2.5 * eta**3 - 1.5 * eta)
                        
                        v6 = scale_x * L3_xi * scale_y * L0_eta
                        v7 = scale_x * L2_xi * scale_y * L1_eta
                        v8 = scale_x * L1_xi * scale_y * L2_eta
                        v9 = scale_x * L0_xi * scale_y * L3_eta
                        val += coeffs[6]*v6 + coeffs[7]*v7 + coeffs[8]*v8 + coeffs[9]*v9
                        
                    idx = t * 3 + k
                    plot_x[idx] = nx
                    plot_y[idx] = ny
                    plot_v[idx] = val

      
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Handle the vlim argument for fixed colorbars
        if vlim is not None:
            vmin, vmax = vlim
        else:
            vmin, vmax = None, None
            
        tc = ax.tripcolor(plot_x, plot_y, plot_tris, plot_v, 
                          shading='gouraud', cmap='inferno',
                          vmin=vmin, vmax=vmax)
        
        faces = self.mesh._cpp_mesh.faces.reshape(-1, 4)
        bnd_faces = self.mesh._cpp_mesh.bnd_faces.reshape(-1, 3)
        
        segments = []
        for f in faces:
            nA, nB = f[2], f[3]
            segments.append([nodes[nA], nodes[nB]])
        for f in bnd_faces:
            nA, nB = f[1], f[2]
            segments.append([nodes[nA], nodes[nB]])
            
        lc = LineCollection(segments, colors='white', linewidths=0.25, zorder=3)
        ax.add_collection(lc)

        plt.colorbar(tc, label='Field Value')
        plt.title('morphdG : Discontinuous Galerkin Solution')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(save_as, dpi=300)
        print(f" -> Saved to '{save_as}'!")
        plt.close()



    def calculate_stable_dt(self, v_max, K_max, cfl=0.1):
        """
        Automatically calculates the maximum stable time step for explicit RK4.
        v_max: Maximum absolute wind speed (e.g., sqrt(vx^2 + vy^2))
        K_max: Maximum diffusion coefficient
        cfl: Safety factor (0.1 to 0.2 is usually safe for DG)
        """
        # 1. Find the smallest physical edge (h_min) in the mesh
        nodes = self.mesh._cpp_mesh.nodes.reshape(-1, 2)
        faces = self.mesh._cpp_mesh.faces.reshape(-1, 4)
        
        # faces[:, 2] and faces[:, 3] hold the node indices for the edge
        nA = faces[:, 2]
        nB = faces[:, 3]
        
        dx = nodes[nA, 0] - nodes[nB, 0]
        dy = nodes[nA, 1] - nodes[nB, 1]
        edge_lengths = np.sqrt(dx**2 + dy**2)
        h_min = np.min(edge_lengths)
        
        # 2. Get the maximum polynomial order currently in use
        p = np.max(self.p_orders)
        
        # 3. Calculate the Advection Limit
        if v_max > 1e-12:
            dt_adv = cfl * (h_min / (v_max * (2 * p + 1)))
        else:
            dt_adv = float('inf')
            
        # 4. Calculate the Diffusion Limit (Includes a rough penalty adjustment)
        if K_max > 1e-12:
            # The penalty parameter (alpha) heavily restricts explicit diffusion
            alpha_penalty = self._cpp_solver.coeffs.alpha
            dt_diff = cfl * (h_min**2 / (2 * K_max * alpha_penalty * (p + 1)**2))
        else:
            dt_diff = float('inf')
            
        # The true safe time step is the smallest of the two limits!
        safe_dt = min(dt_adv, dt_diff)
        
        print(f"--- Stability Analysis ---")
        print(f" -> Min Edge Length (h) : {h_min:.6f}")
        print(f" -> Max Poly Order (p)  : {p}")
        print(f" -> Advection dt limit  : {dt_adv:.8f}")
        print(f" -> Diffusion dt limit  : {dt_diff:.8f}")
        print(f" -> RECOMMENDED dt      : {safe_dt:.8f}")
        
        return safe_dt


    def advance_time_implicit(self, dt):
        """Steps the physics forward by dt using Unconditionally Stable Backward Euler."""
        self._cpp_solver.advance_implicit_euler(dt)
        return self._cpp_solver.get_state()
