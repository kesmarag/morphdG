#pragma once
#include <Kokkos_Core.hpp>
#include <vector>
#include <iomanip>

struct Coeffs {
  // Diffusion Tensor
  // double Kxx = 0.0;
  // double Kxy = 0.0;
  // double Kyx = 0.0;
  // double Kyy = 0.0;

  // Advection Velocity
  // double vx = 0.0;
  // double vy = 0.0;

  // DG Penalty Parameter
  double alpha = 0.0;
};

using Real = double;

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;

constexpr int MAX_BASIS = 6; // P=2 -> 6 basis functions

// Views
using ViewInt1D = Kokkos::View<int *, Kokkos::LayoutLeft, MemorySpace>;
using ViewInt2D = Kokkos::View<int **, Kokkos::LayoutLeft, MemorySpace>;
using ViewReal1D = Kokkos::View<Real *, Kokkos::LayoutLeft, MemorySpace>;
using ViewReal2D = Kokkos::View<Real **, Kokkos::LayoutLeft, MemorySpace>;

// CSR Views
using ViewCSRRows = Kokkos::View<int *, Kokkos::LayoutLeft, MemorySpace>;
using ViewCSRCols = Kokkos::View<int *, Kokkos::LayoutLeft, MemorySpace>;
using ViewCSRVals = Kokkos::View<Real *, Kokkos::LayoutLeft, MemorySpace>;

struct QuadData {
  KOKKOS_INLINE_FUNCTION
  static int get_num_vol_points() { return 4; }

  KOKKOS_INLINE_FUNCTION
  static void get_vol(int q, Real &s, Real &t, Real &w) {
    if (q == 0) {
      s = 1.0 / 3.0;
      t = 1.0 / 3.0;
      w = -0.28125;
    } else if (q == 1) {
      s = 1.0 / 5.0;
      t = 1.0 / 5.0;
      w = 0.2604166666666667;
    } else if (q == 2) {
      s = 3.0 / 5.0;
      t = 1.0 / 5.0;
      w = 0.2604166666666667;
    } else {
      s = 1.0 / 5.0;
      t = 3.0 / 5.0;
      w = 0.2604166666666667;
    }
  }
};

struct QuadData1D {
  KOKKOS_INLINE_FUNCTION
  static int get_num_face_points() { return 2; }

  KOKKOS_INLINE_FUNCTION
  static void get_face(int q, Real &xi, Real &w) {
    if (q == 0) {
      xi = -0.5773502691896257;
      w = 1.0;
    } else {
      xi = 0.5773502691896257;
      w = 1.0;
    }
  }
};

KOKKOS_INLINE_FUNCTION
int get_basis_count(int p) { return (p + 1) * (p + 2) / 2; }

KOKKOS_INLINE_FUNCTION
void get_affine_map(Real p1x, Real p1y, Real p2x, Real p2y, Real p3x, Real p3y,
                    Real &B00, Real &B01, Real &B10, Real &B11, Real &cx,
                    Real &cy) {
  B00 = p2x - p1x;
  B01 = p3x - p1x;
  B10 = p2y - p1y;
  B11 = p3y - p1y;
  cx = p1x;
  cy = p1y;
}

KOKKOS_INLINE_FUNCTION
void legendre_poly(int n, Real x, Real &val, Real &dval) {
  if (n == 0) {
    val = 0.7071067811865475;
    dval = 0.0;
  } else if (n == 1) {
    val = 1.224744871391589 * x;
    dval = 1.224744871391589;
  } else if (n == 2) {
    Real c = 1.5811388300841898;
    val = c * (1.5 * x * x - 0.5);
    dval = c * (3.0 * x);
  }
}

KOKKOS_INLINE_FUNCTION
void eval_physical_basis_dynamic(int basis_idx, Real bb_min_x, Real bb_min_y,
                                 Real bb_max_x, Real bb_max_y, Real x_phys,
                                 Real y_phys, Real &val, Real &grad_x,
                                 Real &grad_y) {
  Real dx = bb_max_x - bb_min_x;
  Real dy = bb_max_y - bb_min_y;

  Real xi = 2.0 * (x_phys - bb_min_x) / dx - 1.0;
  Real eta = 2.0 * (y_phys - bb_min_y) / dy - 1.0;
  Real dxi_dx = (2.0 / dx);
  Real deta_dy = (2.0 / dy);
  Real scale_x = Kokkos::sqrt(2.0 / dx);
  Real scale_y = Kokkos::sqrt(2.0 / dy);

  int nx = 0, ny = 0;
  if (basis_idx == 1) {
    nx = 1;
    ny = 0;
  } else if (basis_idx == 2) {
    nx = 0;
    ny = 1;
  } else if (basis_idx == 3) {
    nx = 2;
    ny = 0;
  } else if (basis_idx == 4) {
    nx = 1;
    ny = 1;
  } else if (basis_idx == 5) {
    nx = 0;
    ny = 2;
  }

  Real Lx, dLx, Ly, dLy;
  legendre_poly(nx, xi, Lx, dLx);
  legendre_poly(ny, eta, Ly, dLy);

  val = (scale_x * Lx) * (scale_y * Ly);
  grad_x = (scale_x * dLx * dxi_dx) * (scale_y * Ly);
  grad_y = (scale_x * Lx) * (scale_y * dLy * deta_dy);
}

KOKKOS_INLINE_FUNCTION
int linear_search(const ViewCSRCols &cols, int start_idx, int end_idx,
                  int target) {
  for (int i = start_idx; i < end_idx; ++i) {
    if (cols(i) == target)
      return i;
  }
  return -1;
}



template <typename RowView, typename ColView, typename ValView,
          typename OffsetView>
void print_csr_matrix_blocks(const RowView &h_row_ptr, const ColView &h_col_ind,
                             const ValView &h_vals, const OffsetView &h_offsets,
                             int print_limit = 20) {

  int num_rows = h_row_ptr.extent(0) - 1;
  int limit = (print_limit < num_rows) ? print_limit : num_rows;

  auto is_boundary = [&](int idx) {
    int num_offsets = h_offsets.extent(0);
    for (int o = 1; o < num_offsets; ++o) {
      if (idx == h_offsets(o))
        return true;
    }
    return false;
  };

  std::cout << "\n--- LHS MATRIX (" << num_rows << "x" << num_rows << ") ---\n";

  for (int i = 0; i < limit; ++i) {
    if (i > 0 && is_boundary(i)) {
      std::cout << "         ";
      for (int j = 0; j < limit; ++j) {
        if (j > 0 && is_boundary(j))
          std::cout << " + ";
        std::cout << "-------";
      }
      std::cout << "\n";
    }

    std::cout << "Row " << std::setw(2) << i << ": [ ";
    for (int j = 0; j < limit; ++j) {
      if (j > 0 && is_boundary(j))
        std::cout << "| ";

      double val = 0.0;
      int row_start = h_row_ptr(i);
      int row_end = h_row_ptr(i + 1);
      for (int k = row_start; k < row_end; ++k) {
        if (h_col_ind(k) == j) {
          val = h_vals(k);
          break;
        }
      }

      if (std::abs(val) < 1e-12)
        std::cout << "   .   ";
      else
        std::cout << std::fixed << std::setprecision(2) << std::setw(6) << val
                  << " ";
    }
    std::cout << "]\n";
  }

  if (limit < num_rows) {
    std::cout << "... (matrix truncated for display. " << num_rows - limit
              << " rows hidden) ...\n";
  }
}


struct AssembleVolumeKernel {
  ViewCSRVals global_vals;
  ViewCSRCols global_cols;
  ViewCSRRows global_rows;
  ViewReal1D rhs_vector;
  ViewInt1D t_offsets;
  ViewInt2D triangles;
  ViewReal2D nodes;
  ViewReal2D bboxes;
  ViewInt1D element_orders;
  ViewInt1D dof_offsets;

  // --- 1. NEW FIELD VIEWS (Replacing old scalars) ---
  ViewReal1D source_exact;
  ViewReal1D vx_quad, vy_quad;
  ViewReal1D Kxx_quad, Kxy_quad, Kyx_quad, Kyy_quad;

  // --- 2. UPDATED CONSTRUCTOR ---
  AssembleVolumeKernel(ViewCSRVals _vals, ViewCSRCols _cols, ViewCSRRows _rows,
                       ViewReal1D _rhs, ViewInt1D _toff, ViewInt2D _tri,
                       ViewReal2D _nodes, ViewReal2D _bb, ViewInt1D _orders,
                       ViewInt1D _offsets, ViewReal1D _source_exact,
                       ViewReal1D _vx, ViewReal1D _vy, ViewReal1D _Kxx,
                       ViewReal1D _Kxy, ViewReal1D _Kyx, ViewReal1D _Kyy)
      : global_vals(_vals), global_cols(_cols), global_rows(_rows),
        rhs_vector(_rhs), t_offsets(_toff), triangles(_tri), nodes(_nodes),
        bboxes(_bb), element_orders(_orders), dof_offsets(_offsets),
        source_exact(_source_exact), vx_quad(_vx), vy_quad(_vy), Kxx_quad(_Kxx),
        Kxy_quad(_Kxy), Kyx_quad(_Kyx), Kyy_quad(_Kyy) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int e) const {
    int p = element_orders(e);
    int n_basis = get_basis_count(p);
    int row_start_global = dof_offsets(e);

    Real bb_min_x = bboxes(e, 0), bb_min_y = bboxes(e, 1);
    Real bb_max_x = bboxes(e, 2), bb_max_y = bboxes(e, 3);

    Real local_Ke[MAX_BASIS][MAX_BASIS] = {0.0};
    Real local_rhs[MAX_BASIS] = {0.0};

    int start = t_offsets(e);
    int end = t_offsets(e + 1);

    for (int t_idx = start; t_idx < end; ++t_idx) {
      int n1 = triangles(t_idx, 0), n2 = triangles(t_idx, 1),
          n3 = triangles(t_idx, 2);

      Real B00, B01, B10, B11, cx, cy;
      get_affine_map(nodes(n1, 0), nodes(n1, 1), nodes(n2, 0), nodes(n2, 1),
                     nodes(n3, 0), nodes(n3, 1), B00, B01, B10, B11, cx, cy);
      Real detJ = Kokkos::abs(B00 * B11 - B01 * B10);

      int n_q = QuadData::get_num_vol_points();
      for (int q = 0; q < n_q; ++q) {
        Real qr_s, qr_t, qr_w;
        QuadData::get_vol(q, qr_s, qr_t, qr_w);

        Real x_phys = B00 * qr_s + B01 * qr_t + cx;
        Real y_phys = B10 * qr_s + B11 * qr_t + cy;
        Real w = qr_w * detJ;

        Real vals[MAX_BASIS], gx[MAX_BASIS], gy[MAX_BASIS];
        for (int i = 0; i < n_basis; ++i) {
          eval_physical_basis_dynamic(i, bb_min_x, bb_min_y, bb_max_x, bb_max_y,
                                      x_phys, y_phys, vals[i], gx[i], gy[i]);
        }

        // --- 3. THE MAGIC: O(1) MEMORY LOOKUP INSTEAD OF SCALARS ---
        // Calculate exactly which quadrature point we are at globally
        int global_q_idx = t_idx * n_q + q;

        // Read physics directly from Kokkos GPU memory
        Real source_f = source_exact(global_q_idx);
        Real l_vx = vx_quad(global_q_idx);
        Real l_vy = vy_quad(global_q_idx);
        Real l_Kxx = Kxx_quad(global_q_idx);
        Real l_Kxy = Kxy_quad(global_q_idx);
        Real l_Kyx = Kyx_quad(global_q_idx);
        Real l_Kyy = Kyy_quad(global_q_idx);

        for (int r = 0; r < n_basis; ++r) {
          local_rhs[r] += source_f * vals[r] * w;

          for (int c = 0; c < n_basis; c++) {
            // Plug the local variables right into your exact same math!
            Real diff_term = gx[r] * (l_Kxx * gx[c] + l_Kxy * gy[c]) +
                             gy[r] * (l_Kyx * gx[c] + l_Kyy * gy[c]);

            Real adv_term = (l_vx * gx[r] + l_vy * gy[r]) * vals[c];

            local_Ke[r][c] += (diff_term - adv_term) * w;
          }
        }
      }
    }

    // --- SCATTER TO GLOBAL MATRIX & VECTOR (Unchanged) ---
    for (int r = 0; r < n_basis; ++r) {
      int global_row = row_start_global + r;

      if (Kokkos::abs(local_rhs[r]) > 1e-14) {
        Kokkos::atomic_add(&rhs_vector(global_row), local_rhs[r]);
      }

      int row_ptr_start = global_rows(global_row);
      int row_ptr_end = global_rows(global_row + 1);

      for (int c = 0; c < n_basis; c++) {
        if (Kokkos::abs(local_Ke[r][c]) > 1e-12) {
          int global_col = row_start_global + c;
          int idx = linear_search(global_cols, row_ptr_start, row_ptr_end,
                                  global_col);
          if (idx != -1)
            Kokkos::atomic_add(&global_vals(idx), local_Ke[r][c]);
        }
      }
    }
  }
};

struct DGSolver {

  Coeffs coeffs;
  int total_dofs = 0; // Total size of the global matrix

  // --- NODAL VIEWS ---
  Kokkos::View<double *, Kokkos::LayoutLeft,
               Kokkos::DefaultExecutionSpace::memory_space>
      d_source_nodal;

  void set_source_nodal(const double *data, int n) {
    Kokkos::View<const double *, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        h_raw(data, n);
    Kokkos::resize(d_source_nodal, n);
    Kokkos::deep_copy(d_source_nodal, h_raw);
  }

  Kokkos::View<double *, Kokkos::LayoutLeft,
               Kokkos::DefaultExecutionSpace::memory_space>
      d_boundary_nodal;

  DGSolver() = default;

  // --- EXACT QUADRATURE COORDINATE GENERATOR ---
  std::pair<std::vector<double>, std::vector<double>>
  vol_quad_points(const AggMesh &mesh) {
    int n_tri = mesh.num_triangles();
    int n_q = QuadData::get_num_vol_points();

    std::vector<double> x_quads(n_tri * n_q);
    std::vector<double> y_quads(n_tri * n_q);

    for (int t = 0; t < n_tri; ++t) {
      int n1 = mesh.h_triangles[3 * t + 0];
      int n2 = mesh.h_triangles[3 * t + 1];
      int n3 = mesh.h_triangles[3 * t + 2];

      double B00, B01, B10, B11, cx, cy;
      // Now it cleanly calls the global get_affine_map from the top of the
      // file!
      get_affine_map(mesh.h_nodes[2 * n1], mesh.h_nodes[2 * n1 + 1],
                     mesh.h_nodes[2 * n2], mesh.h_nodes[2 * n2 + 1],
                     mesh.h_nodes[2 * n3], mesh.h_nodes[2 * n3 + 1], B00, B01,
                     B10, B11, cx, cy);

      for (int q = 0; q < n_q; ++q) {
        double s, t_coord, w;
        QuadData::get_vol(q, s, t_coord, w);

        x_quads[t * n_q + q] = B00 * s + B01 * t_coord + cx;
        y_quads[t * n_q + q] = B10 * s + B11 * t_coord + cy;
      }
    }
    return {x_quads, y_quads};
  }

  // --- ADAPTIVITY STATE VIEWS ---
  Kokkos::View<int *, Kokkos::LayoutLeft,
               Kokkos::DefaultExecutionSpace::memory_space>
      d_orders;
  Kokkos::View<int *, Kokkos::LayoutLeft,
               Kokkos::DefaultExecutionSpace::memory_space>
      d_offsets;

  void set_p_orders(const int *data, int n_elem) {
    Kokkos::View<const int *, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        h_orders_raw(data, n_elem);
    Kokkos::resize(d_orders, n_elem);
    Kokkos::deep_copy(d_orders, h_orders_raw);

    Kokkos::resize(d_offsets, n_elem + 1);
    auto h_offsets = Kokkos::create_mirror_view(d_offsets);
    h_offsets(0) = 0;

    for (int i = 0; i < n_elem; ++i) {
      // Cleanly calls the global get_basis_count!
      h_offsets(i + 1) = h_offsets(i) + get_basis_count(h_orders_raw(i));
    }
    Kokkos::deep_copy(d_offsets, h_offsets);
    total_dofs = h_offsets(n_elem);
  }

  // --- VELOCITY FIELDS ---
  Kokkos::View<double *, Kokkos::LayoutLeft,
               Kokkos::DefaultExecutionSpace::memory_space>
      d_vx_quad;
  Kokkos::View<double *, Kokkos::LayoutLeft,
               Kokkos::DefaultExecutionSpace::memory_space>
      d_vy_quad;

  void set_vx_field(const double *data, int n) {
    Kokkos::View<const double *, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        h_raw(data, n);
    Kokkos::resize(d_vx_quad, n);
    Kokkos::deep_copy(d_vx_quad, h_raw);
  }

  void set_vy_field(const double *data, int n) {
    Kokkos::View<const double *, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        h_raw(data, n);
    Kokkos::resize(d_vy_quad, n);
    Kokkos::deep_copy(d_vy_quad, h_raw);
  }

  // --- DIFFUSION TENSOR FIELDS ---
  Kokkos::View<double *, Kokkos::LayoutLeft,
               Kokkos::DefaultExecutionSpace::memory_space>
      d_Kxx_quad;
  Kokkos::View<double *, Kokkos::LayoutLeft,
               Kokkos::DefaultExecutionSpace::memory_space>
      d_Kxy_quad;
  Kokkos::View<double *, Kokkos::LayoutLeft,
               Kokkos::DefaultExecutionSpace::memory_space>
      d_Kyx_quad;
  Kokkos::View<double *, Kokkos::LayoutLeft,
               Kokkos::DefaultExecutionSpace::memory_space>
      d_Kyy_quad;

  void set_Kxx_field(const double *data, int n) {
    Kokkos::View<const double *, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        h_raw(data, n);
    Kokkos::resize(d_Kxx_quad, n);
    Kokkos::deep_copy(d_Kxx_quad, h_raw);
  }
  void set_Kxy_field(const double *data, int n) {
    Kokkos::View<const double *, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        h_raw(data, n);
    Kokkos::resize(d_Kxy_quad, n);
    Kokkos::deep_copy(d_Kxy_quad, h_raw);
  }
  void set_Kyx_field(const double *data, int n) {
    Kokkos::View<const double *, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        h_raw(data, n);
    Kokkos::resize(d_Kyx_quad, n);
    Kokkos::deep_copy(d_Kyx_quad, h_raw);
  }
  void set_Kyy_field(const double *data, int n) {
    Kokkos::View<const double *, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        h_raw(data, n);
    Kokkos::resize(d_Kyy_quad, n);
    Kokkos::deep_copy(d_Kyy_quad, h_raw);
  }

  // --- GLOBAL SYSTEM MEMORY ---
  ViewCSRVals d_global_vals;
  ViewCSRCols d_global_cols;
  ViewCSRRows d_global_rows;
  ViewReal1D d_rhs;

  // --- KOKKOS KERNEL LAUNCHER ---
  void assemble_volume(const AggMesh &mesh) {
    // 1. Zero out the matrix values and RHS vector before accumulating
    // (We use Kokkos::deep_copy to blast zeros across the GPU memory instantly)
    Kokkos::deep_copy(d_global_vals, 0.0);
    Kokkos::deep_copy(d_rhs, 0.0);

    // 2. Instantiate the Functor with all your zero-copy views
    AssembleVolumeKernel vol_kernel(
        d_global_vals, d_global_cols, d_global_rows, d_rhs, mesh.d_t_offsets,
        mesh.d_triangles, mesh.d_nodes, mesh.d_bboxes, // Mesh data
        d_orders, d_offsets,                           // DG data
        d_source_nodal,                                // Source field
        d_vx_quad, d_vy_quad,                          // Velocity fields
        d_Kxx_quad, d_Kxy_quad, d_Kyx_quad, d_Kyy_quad // Tensor fields
    );

    // 3. LAUNCH THE GPU KERNEL!
    // We launch 1 thread per polygonal element (mesh.n_elems).
    Kokkos::parallel_for("AssembleVolume", mesh.num_elements(), vol_kernel);

    // 4. Synchronization barrier: Wait for the GPU to finish before returning
    // to Python
    Kokkos::fence();
  }

  // Inside struct DGSolver in include/morphdg/dg_solver.hpp

  // --- SPARSE MATRIX ALLOCATOR ---
  void create_sparse_graph(const AggMesh &mesh) {
    if (total_dofs == 0) {
      throw std::runtime_error(
          "You must call set_p_orders() before creating the sparse graph!");
    }

    int n_elem = mesh.num_elements();

    // 1. Build an element-to-neighbor adjacency list using the mesh faces
    std::vector<std::vector<int>> elem_neighbors(n_elem);
    int num_faces = mesh.h_faces.size() / 4;
    for (int f = 0; f < num_faces; ++f) {
      int ePlus = mesh.h_faces[4 * f + 0];
      int eMinus = mesh.h_faces[4 * f + 1];
      // Connect the two elements sharing this face
      elem_neighbors[ePlus].push_back(eMinus);
      elem_neighbors[eMinus].push_back(ePlus);
    }

    // 2. Fetch the DOF offsets from the GPU back to the CPU temporarily
    auto h_offsets =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_offsets);

    // 3. Build the Sparsity Pattern (List of non-zero columns for each global
    // row)
    std::vector<std::vector<int>> row_cols(total_dofs);

    for (int e = 0; e < n_elem; ++e) {
      int row_start = h_offsets(e);
      int row_end = h_offsets(e + 1);
      int n_basis_e = row_end - row_start;

      // Gather all elements that interact with element e (itself + neighbors)
      std::vector<int> coupled_elements = {e};
      for (int neighbor : elem_neighbors[e]) {
        coupled_elements.push_back(neighbor);
      }

      // Map out the blocks for every global row belonging to element e
      for (int r = 0; r < n_basis_e; ++r) {
        int global_row = row_start + r;

        // Add all DOFs from all coupled elements to this row
        for (int c_elem : coupled_elements) {
          int col_start = h_offsets(c_elem);
          int col_end = h_offsets(c_elem + 1);
          for (int c = col_start; c < col_end; ++c) {
            row_cols[global_row].push_back(c);
          }
        }

        // Sort the columns (Required for standard CSR format)
        std::sort(row_cols[global_row].begin(), row_cols[global_row].end());
      }
    }

    // 4. Flatten the graph into the CSR 1D arrays
    std::vector<int> h_rows(total_dofs + 1, 0);
    std::vector<int> h_cols;
    int nnz = 0; // Number of Non-Zeros

    for (int i = 0; i < total_dofs; ++i) {
      h_rows[i] = nnz;
      for (int col : row_cols[i]) {
        h_cols.push_back(col);
        nnz++;
      }
    }
    h_rows[total_dofs] = nnz;

    // 5. Allocate the Kokkos GPU Views!
    d_global_rows = ViewCSRRows("d_global_rows", total_dofs + 1);
    d_global_cols = ViewCSRCols("d_global_cols", nnz);
    d_global_vals = ViewCSRVals("d_global_vals", nnz); // Matrix values
    d_rhs = ViewReal1D("d_rhs", total_dofs);           // RHS vector

    // 6. Deep copy the graph structure to the GPU
    auto mirror_rows = Kokkos::create_mirror_view(d_global_rows);
    auto mirror_cols = Kokkos::create_mirror_view(d_global_cols);

    for (int i = 0; i <= total_dofs; ++i)
      mirror_rows(i) = h_rows[i];
    for (int i = 0; i < nnz; ++i)
      mirror_cols(i) = h_cols[i];

    Kokkos::deep_copy(d_global_rows, mirror_rows);
    Kokkos::deep_copy(d_global_cols, mirror_cols);

    // (Note: d_global_vals and d_rhs are filled with 0.0 automatically inside
    // assemble_volume!)
  }

// Inside struct DGSolver
  
  void print_matrix(int print_limit = 20) {
    if (total_dofs == 0 || d_global_rows.extent(0) == 0) {
      std::cout << "Matrix is empty or not allocated yet!\n";
      return;
    }

    // 1. Safely pull the CSR arrays from the GPU back to the CPU
    auto h_rows = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_global_rows);
    auto h_cols = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_global_cols);
    auto h_vals = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_global_vals);
    auto h_offs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_offsets);

    // 2. Call your print function using the Host Views
    print_csr_matrix_blocks(h_rows, h_cols, h_vals, h_offs, print_limit);
  }


  
};
