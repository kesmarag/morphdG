#pragma once
#include <KokkosBlas1_axpby.hpp>
#include <KokkosBlas1_dot.hpp>
#include <KokkosBlas1_nrm2.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spmv.hpp>
#include <Kokkos_Core.hpp>
#include <iomanip>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

struct Coeffs {
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

        // Calculate exactly which quadrature point we are at globally
        int global_q_idx = t_idx * n_q + q;

        // Read directly from Kokkos DEVICE memory
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
            Real diff_term = gx[r] * (l_Kxx * gx[c] + l_Kxy * gy[c]) +
                             gy[r] * (l_Kyx * gx[c] + l_Kyy * gy[c]);

            Real adv_term = (l_vx * gx[r] + l_vy * gy[r]) * vals[c];

            local_Ke[r][c] += (diff_term - adv_term) * w;
          }
        }
      }
    }

    // --- SCATTER TO GLOBAL MATRIX & VECTOR ---
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

struct AssembleFaceKernel {
  ViewCSRVals global_vals;
  ViewCSRCols global_cols;
  ViewCSRRows global_rows;
  ViewInt1D faces;
  ViewReal2D nodes;
  ViewReal2D bboxes;
  ViewInt1D element_orders;
  ViewInt1D dof_offsets;

  ViewReal1D vx_face, vy_face;
  ViewReal1D Kxx_face, Kxy_face, Kyx_face, Kyy_face;
  Real alpha; // The penalty parameter

  AssembleFaceKernel(ViewCSRVals _vals, ViewCSRCols _cols, ViewCSRRows _rows,
                     ViewInt1D _faces, ViewReal2D _nodes, ViewReal2D _bb,
                     ViewInt1D _orders, ViewInt1D _offsets, ViewReal1D _vxf,
                     ViewReal1D _vyf, ViewReal1D _Kxxf, ViewReal1D _Kxyf,
                     ViewReal1D _Kyxf, ViewReal1D _Kyyf, Real _alpha)
      : global_vals(_vals), global_cols(_cols), global_rows(_rows),
        faces(_faces), nodes(_nodes), bboxes(_bb), element_orders(_orders),
        dof_offsets(_offsets), vx_face(_vxf), vy_face(_vyf), Kxx_face(_Kxxf),
        Kxy_face(_Kxyf), Kyx_face(_Kyxf), Kyy_face(_Kyyf), alpha(_alpha) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int f) const {
    // Unpack Face Geometry
    int ePlus = faces(4 * f + 0);
    int eMinus = faces(4 * f + 1);
    int nA = faces(4 * f + 2);
    int nB = faces(4 * f + 3);

    Real xA = nodes(nA, 0), yA = nodes(nA, 1);
    Real xB = nodes(nB, 0), yB = nodes(nB, 1);
    Real dx = xB - xA, dy = yB - yA;
    Real h_edge = Kokkos::sqrt(dx * dx + dy * dy);

    // Unoriented Normal Vector
    Real nx = dy / h_edge;
    Real ny = -dx / h_edge;

    // Ensure Normal points OUTWARD from ePlus (+)
    Real cx = 0.5 * (bboxes(ePlus, 0) + bboxes(ePlus, 2));
    Real cy = 0.5 * (bboxes(ePlus, 1) + bboxes(ePlus, 3));
    Real face_cx = 0.5 * (xA + xB), face_cy = 0.5 * (yA + yB);
    if ((nx * (face_cx - cx) + ny * (face_cy - cy)) < 0.0) {
      nx = -nx;
      ny = -ny;
    }

    // 2. Element Setup
    int pPlus = element_orders(ePlus), pMinus = element_orders(eMinus);
    int n_bPlus = get_basis_count(pPlus), n_bMinus = get_basis_count(pMinus);
    int row_start_Plus = dof_offsets(ePlus),
        row_start_Minus = dof_offsets(eMinus);

    // Local 4x4 Interaction Blocks
    Real K_PP[MAX_BASIS][MAX_BASIS] = {0.0}; // Plus-Plus
    Real K_PM[MAX_BASIS][MAX_BASIS] = {0.0}; // Plus-Minus
    Real K_MP[MAX_BASIS][MAX_BASIS] = {0.0}; // Minus-Plus
    Real K_MM[MAX_BASIS][MAX_BASIS] = {0.0}; // Minus-Minus

    int n_q = QuadData1D::get_num_face_points();

    for (int q = 0; q < n_q; ++q) {
      Real xi, w_ref;
      QuadData1D::get_face(q, xi, w_ref);
      Real w = w_ref * (h_edge / 2.0); // Jacobian of 1D line

      // Physical coordinates of quad point
      Real shapeA = 0.5 * (1.0 - xi), shapeB = 0.5 * (1.0 + xi);
      Real x_phys = shapeA * xA + shapeB * xB;
      Real y_phys = shapeA * yA + shapeB * yB;

      int global_q_idx = f * n_q + q;
      Real l_vx = vx_face(global_q_idx), l_vy = vy_face(global_q_idx);
      Real l_Kxx = Kxx_face(global_q_idx), l_Kyy = Kyy_face(global_q_idx);

      // Calculate Upwind Physics
      Real vn = l_vx * nx + l_vy * ny;            // Normal Velocity
      Real upwind_flux_P = (vn > 0.0) ? vn : 0.0; // Leaves Plus
      Real upwind_flux_M = (vn < 0.0) ? vn : 0.0; // Leaves Minus

      // SIPG Penalty Parameter (using max polynomial and local diffusion)
      Real p_max = (pPlus > pMinus) ? pPlus : pMinus;
      Real K_max =
          (l_Kxx > l_Kyy) ? l_Kxx : l_Kyy; // Simplified tensor magnitude
      Real penalty = alpha * (p_max * p_max / h_edge) * K_max;

      // Evaluate Basis Functions at the edge for both elements
      Real vP[MAX_BASIS], gxP[MAX_BASIS], gyP[MAX_BASIS];
      Real vM[MAX_BASIS], gxM[MAX_BASIS], gyM[MAX_BASIS];

      for (int i = 0; i < n_bPlus; ++i)
        eval_physical_basis_dynamic(i, bboxes(ePlus, 0), bboxes(ePlus, 1),
                                    bboxes(ePlus, 2), bboxes(ePlus, 3), x_phys,
                                    y_phys, vP[i], gxP[i], gyP[i]);

      for (int i = 0; i < n_bMinus; ++i)
        eval_physical_basis_dynamic(i, bboxes(eMinus, 0), bboxes(eMinus, 1),
                                    bboxes(eMinus, 2), bboxes(eMinus, 3),
                                    x_phys, y_phys, vM[i], gxM[i], gyM[i]);

      // --- THE 4 MATRIX BLOCKS ---
      for (int r = 0; r < n_bPlus; ++r) {
        for (int c = 0; c < n_bPlus; ++c) { // Plus-Plus
          Real adv = upwind_flux_P * vP[r] * vP[c];
          Real diff =
              penalty * vP[r] * vP[c] -
              0.5 * (l_Kxx * gxP[c] * nx + l_Kyy * gyP[c] * ny) * vP[r] -
              0.5 * (l_Kxx * gxP[r] * nx + l_Kyy * gyP[r] * ny) * vP[c];
          K_PP[r][c] += (adv + diff) * w;
        }
        for (int c = 0; c < n_bMinus; ++c) { // Plus-Minus
          Real adv = upwind_flux_M * vP[r] * vM[c];
          Real diff =
              -penalty * vP[r] * vM[c] -
              0.5 * (l_Kxx * gxM[c] * nx + l_Kyy * gyM[c] * ny) * vP[r] +
              0.5 * (l_Kxx * gxP[r] * nx + l_Kyy * gyP[r] * ny) * vM[c];
          K_PM[r][c] += (adv + diff) * w;
        }
      }

      for (int r = 0; r < n_bMinus; ++r) {
        for (int c = 0; c < n_bPlus; ++c) { // Minus-Plus
          Real adv = -upwind_flux_P * vM[r] * vP[c];
          Real diff =
              -penalty * vM[r] * vP[c] +
              0.5 * (l_Kxx * gxP[c] * nx + l_Kyy * gyP[c] * ny) * vM[r] -
              0.5 * (l_Kxx * gxM[r] * nx + l_Kyy * gyM[r] * ny) * vP[c];
          K_MP[r][c] += (adv + diff) * w;
        }
        for (int c = 0; c < n_bMinus; ++c) { // Minus-Minus
          Real adv = -upwind_flux_M * vM[r] * vM[c];
          Real diff =
              penalty * vM[r] * vM[c] +
              0.5 * (l_Kxx * gxM[c] * nx + l_Kyy * gyM[c] * ny) * vM[r] +
              0.5 * (l_Kxx * gxM[r] * nx + l_Kyy * gyM[r] * ny) * vM[c];
          K_MM[r][c] += (adv + diff) * w;
        }
      }
    }

    // 3. Scatter All 4 Blocks to Global CSR Matrix
    auto scatter = [&](int r_start, int c_start, int num_r, int num_c,
                       Real local_K[MAX_BASIS][MAX_BASIS]) {
      for (int r = 0; r < num_r; ++r) {
        int global_row = r_start + r;
        int row_ptr_start = global_rows(global_row);
        int row_ptr_end = global_rows(global_row + 1);
        for (int c = 0; c < num_c; ++c) {
          if (Kokkos::abs(local_K[r][c]) > 1e-12) {
            int global_col = c_start + c;
            int idx = linear_search(global_cols, row_ptr_start, row_ptr_end,
                                    global_col);
            if (idx != -1)
              Kokkos::atomic_add(&global_vals(idx), local_K[r][c]);
          }
        }
      }
    };

    scatter(row_start_Plus, row_start_Plus, n_bPlus, n_bPlus, K_PP);
    scatter(row_start_Plus, row_start_Minus, n_bPlus, n_bMinus, K_PM);
    scatter(row_start_Minus, row_start_Plus, n_bMinus, n_bPlus, K_MP);
    scatter(row_start_Minus, row_start_Minus, n_bMinus, n_bMinus, K_MM);
  }
};

struct AssembleBoundaryKernel {
  ViewCSRVals global_vals;
  ViewCSRCols global_cols;
  ViewCSRRows global_rows;
  ViewReal1D rhs_vector;
  ViewInt1D bnd_faces;
  ViewReal2D nodes;
  ViewReal2D bboxes;
  ViewInt1D element_orders;
  ViewInt1D dof_offsets;
  ViewReal1D vx_face, vy_face, Kxx_face, Kyy_face;
  ViewReal1D g_D_face, g_N_face, bctype_face;
  Real alpha;
  int offset_internal_faces;

  AssembleBoundaryKernel(ViewCSRVals _vals, ViewCSRCols _cols,
                         ViewCSRRows _rows, ViewReal1D _rhs,
                         ViewInt1D _bnd_faces, ViewReal2D _nodes,
                         ViewReal2D _bb, ViewInt1D _orders, ViewInt1D _offsets,
                         ViewReal1D _vxf, ViewReal1D _vyf, ViewReal1D _Kxxf,
                         ViewReal1D _Kyyf, ViewReal1D _g_D_face,
                         ViewReal1D _g_N_face, ViewReal1D _bctype_face,
                         Real _alpha, int _offset_internal)
      : global_vals(_vals), global_cols(_cols), global_rows(_rows),
        rhs_vector(_rhs), bnd_faces(_bnd_faces), nodes(_nodes), bboxes(_bb),
        element_orders(_orders), dof_offsets(_offsets), vx_face(_vxf),
        vy_face(_vyf), Kxx_face(_Kxxf), Kyy_face(_Kyyf), g_D_face(_g_D_face),
        g_N_face(_g_N_face), bctype_face(_bctype_face), alpha(_alpha),
        offset_internal_faces(_offset_internal) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int f) const {
    int ePlus = bnd_faces(3 * f + 0), nA = bnd_faces(3 * f + 1),
        nB = bnd_faces(3 * f + 2);
    Real xA = nodes(nA, 0), yA = nodes(nA, 1), xB = nodes(nB, 0),
         yB = nodes(nB, 1);
    Real dx = xB - xA, dy = yB - yA;
    Real h_edge = Kokkos::sqrt(dx * dx + dy * dy);
    Real nx = dy / h_edge, ny = -dx / h_edge;

    Real cx = 0.5 * (bboxes(ePlus, 0) + bboxes(ePlus, 2)),
         cy = 0.5 * (bboxes(ePlus, 1) + bboxes(ePlus, 3));
    if ((nx * (0.5 * (xA + xB) - cx) + ny * (0.5 * (yA + yB) - cy)) < 0.0) {
      nx = -nx;
      ny = -ny;
    }

    int pPlus = element_orders(ePlus), n_bPlus = get_basis_count(pPlus),
        row_start = dof_offsets(ePlus);
    Real local_K[MAX_BASIS][MAX_BASIS] = {0.0};
    Real local_rhs[MAX_BASIS] = {0.0};
    int n_q = QuadData1D::get_num_face_points();

    for (int q = 0; q < n_q; ++q) {
      Real xi, w_ref;
      QuadData1D::get_face(q, xi, w_ref);
      Real w = w_ref * (h_edge / 2.0);
      Real x_phys = 0.5 * (1.0 - xi) * xA + 0.5 * (1.0 + xi) * xB;
      Real y_phys = 0.5 * (1.0 - xi) * yA + 0.5 * (1.0 + xi) * yB;

      int global_q_idx = (offset_internal_faces + f) * n_q + q;
      Real l_vx = vx_face(global_q_idx), l_vy = vy_face(global_q_idx);
      Real l_Kxx = Kxx_face(global_q_idx), l_Kyy = Kyy_face(global_q_idx);

      int bnd_q_idx = f * n_q + q;
      Real l_g_D = g_D_face(bnd_q_idx);
      Real l_g_N = g_N_face(bnd_q_idx);
      Real bc_type = bctype_face(bnd_q_idx);

      Real vn = l_vx * nx + l_vy * ny;
      Real penalty =
          alpha * (pPlus * pPlus / h_edge) * ((l_Kxx > l_Kyy) ? l_Kxx : l_Kyy);

      Real vP[MAX_BASIS], gxP[MAX_BASIS], gyP[MAX_BASIS];
      for (int i = 0; i < n_bPlus; ++i)
        eval_physical_basis_dynamic(i, bboxes(ePlus, 0), bboxes(ePlus, 1),
                                    bboxes(ePlus, 2), bboxes(ePlus, 3), x_phys,
                                    y_phys, vP[i], gxP[i], gyP[i]);

      Real outflow_flux = (vn > 0.0) ? vn : 0.0;
      Real inflow_flux = (vn < 0.0) ? vn : 0.0;

      for (int r = 0; r < n_bPlus; ++r) {
        if (bc_type > 0.5) { // NEUMANN
          local_rhs[r] += (-l_g_N * vP[r]) * w;
          for (int c = 0; c < n_bPlus; ++c)
            local_K[r][c] += (outflow_flux * vP[r] * vP[c]) * w;
        } else { // DIRICHLET
          Real adv_rhs = -inflow_flux * l_g_D * vP[r];
          Real diff_rhs = penalty * l_g_D * vP[r] -
                          (l_Kxx * gxP[r] * nx + l_Kyy * gyP[r] * ny) * l_g_D;
          local_rhs[r] += (adv_rhs + diff_rhs) * w;
          for (int c = 0; c < n_bPlus; ++c) {
            Real adv_lhs = outflow_flux * vP[r] * vP[c];
            Real diff_lhs =
                penalty * vP[r] * vP[c] -
                (l_Kxx * gxP[c] * nx + l_Kyy * gyP[c] * ny) * vP[r] -
                (l_Kxx * gxP[r] * nx + l_Kyy * gyP[r] * ny) * vP[c];
            local_K[r][c] += (adv_lhs + diff_lhs) * w;
          }
        }
      }
    }

    for (int r = 0; r < n_bPlus; ++r) {
      int global_row = row_start + r;
      if (Kokkos::abs(local_rhs[r]) > 1e-14)
        Kokkos::atomic_add(&rhs_vector(global_row), local_rhs[r]);
      int r_start = global_rows(global_row),
          r_end = global_rows(global_row + 1);
      for (int c = 0; c < n_bPlus; ++c) {
        if (Kokkos::abs(local_K[r][c]) > 1e-12) {
          int idx = linear_search(global_cols, r_start, r_end, row_start + c);
          if (idx != -1)
            Kokkos::atomic_add(&global_vals(idx), local_K[r][c]);
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

  // --- EXACT FACE QUADRATURE COORDINATE GENERATOR ---
  std::pair<std::vector<double>, std::vector<double>>
  face_quad_points(const AggMesh &mesh) {
    int n_q_face =
        QuadData1D::get_num_face_points(); // Should be 2 points per face

    int num_int_faces = mesh.h_faces.size() / 4;
    int num_bnd_faces = mesh.h_bnd_faces.size() / 3;
    int total_faces = num_int_faces + num_bnd_faces;

    std::vector<double> x_face(total_faces * n_q_face);
    std::vector<double> y_face(total_faces * n_q_face);

    int offset = 0;

    // Map Internal Faces
    for (int f = 0; f < num_int_faces; ++f) {
      int nA = mesh.h_faces[4 * f + 2];
      int nB = mesh.h_faces[4 * f + 3];

      double xA = mesh.h_nodes[2 * nA], yA = mesh.h_nodes[2 * nA + 1];
      double xB = mesh.h_nodes[2 * nB], yB = mesh.h_nodes[2 * nB + 1];

      for (int q = 0; q < n_q_face; ++q) {
        double xi, w;
        QuadData1D::get_face(q, xi, w);

        // Map from 1D reference [-1, 1] to the physical 2D segment
        double shapeA = 0.5 * (1.0 - xi);
        double shapeB = 0.5 * (1.0 + xi);

        x_face[offset] = shapeA * xA + shapeB * xB;
        y_face[offset] = shapeA * yA + shapeB * yB;
        offset++;
      }
    }

    for (int f = 0; f < num_bnd_faces; ++f) {
      int nA = mesh.h_bnd_faces[3 * f + 1];
      int nB = mesh.h_bnd_faces[3 * f + 2];

      double xA = mesh.h_nodes[2 * nA], yA = mesh.h_nodes[2 * nA + 1];
      double xB = mesh.h_nodes[2 * nB], yB = mesh.h_nodes[2 * nB + 1];

      for (int q = 0; q < n_q_face; ++q) {
        double xi, w;
        QuadData1D::get_face(q, xi, w);

        double shapeA = 0.5 * (1.0 - xi);
        double shapeB = 0.5 * (1.0 + xi);

        x_face[offset] = shapeA * xA + shapeB * xB;
        y_face[offset] = shapeA * yA + shapeB * yB;
        offset++;
      }
    }

    return {x_face, y_face};
  }

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
      h_offsets(i + 1) = h_offsets(i) + get_basis_count(h_orders_raw(i));
    }
    Kokkos::deep_copy(d_offsets, h_offsets);
    total_dofs = h_offsets(n_elem);
  }

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

  Kokkos::View<double *, Kokkos::LayoutLeft,
               Kokkos::DefaultExecutionSpace::memory_space>
      d_vx_face;
  Kokkos::View<double *, Kokkos::LayoutLeft,
               Kokkos::DefaultExecutionSpace::memory_space>
      d_vy_face;

  Kokkos::View<double *, Kokkos::LayoutLeft,
               Kokkos::DefaultExecutionSpace::memory_space>
      d_Kxx_face;
  Kokkos::View<double *, Kokkos::LayoutLeft,
               Kokkos::DefaultExecutionSpace::memory_space>
      d_Kxy_face;
  Kokkos::View<double *, Kokkos::LayoutLeft,
               Kokkos::DefaultExecutionSpace::memory_space>
      d_Kyx_face;
  Kokkos::View<double *, Kokkos::LayoutLeft,
               Kokkos::DefaultExecutionSpace::memory_space>
      d_Kyy_face;

  void set_vx_face(const double *data, int n) {
    Kokkos::View<const double *, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        h_raw(data, n);
    Kokkos::resize(d_vx_face, n);
    Kokkos::deep_copy(d_vx_face, h_raw);
  }
  void set_vy_face(const double *data, int n) {
    Kokkos::View<const double *, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        h_raw(data, n);
    Kokkos::resize(d_vy_face, n);
    Kokkos::deep_copy(d_vy_face, h_raw);
  }

  void set_Kxx_face(const double *data, int n) {
    Kokkos::View<const double *, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        h_raw(data, n);
    Kokkos::resize(d_Kxx_face, n);
    Kokkos::deep_copy(d_Kxx_face, h_raw);
  }
  void set_Kxy_face(const double *data, int n) {
    Kokkos::View<const double *, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        h_raw(data, n);
    Kokkos::resize(d_Kxy_face, n);
    Kokkos::deep_copy(d_Kxy_face, h_raw);
  }
  void set_Kyx_face(const double *data, int n) {
    Kokkos::View<const double *, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        h_raw(data, n);
    Kokkos::resize(d_Kyx_face, n);
    Kokkos::deep_copy(d_Kyx_face, h_raw);
  }
  void set_Kyy_face(const double *data, int n) {
    Kokkos::View<const double *, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        h_raw(data, n);
    Kokkos::resize(d_Kyy_face, n);
    Kokkos::deep_copy(d_Kyy_face, h_raw);
  }
  
  Kokkos::View<double *, Kokkos::LayoutLeft,
               Kokkos::DefaultExecutionSpace::memory_space>
      d_g_D_face;
  Kokkos::View<double *, Kokkos::LayoutLeft,
               Kokkos::DefaultExecutionSpace::memory_space>
      d_g_N_face;
  Kokkos::View<double *, Kokkos::LayoutLeft,
               Kokkos::DefaultExecutionSpace::memory_space>
      d_bctype_face;

  void set_g_D_face(const double *data, int n) {
    Kokkos::View<const double *, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        h_raw(data, n);
    Kokkos::resize(d_g_D_face, n);
    Kokkos::deep_copy(d_g_D_face, h_raw);
  }
  void set_g_N_face(const double *data, int n) {
    Kokkos::View<const double *, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        h_raw(data, n);
    Kokkos::resize(d_g_N_face, n);
    Kokkos::deep_copy(d_g_N_face, h_raw);
  }
  void set_bctype_face(const double *data, int n) {
    Kokkos::View<const double *, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        h_raw(data, n);
    Kokkos::resize(d_bctype_face, n);
    Kokkos::deep_copy(d_bctype_face, h_raw);
  }

  // --- PYTHON MATRIX EXTRACTOR ---
  pybind11::tuple get_global_system() {
    auto h_vals =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_global_vals);
    auto h_cols =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_global_cols);
    auto h_rows =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_global_rows);
    auto h_rhs =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_rhs);

    pybind11::array_t<double> py_vals(h_vals.extent(0), h_vals.data());
    pybind11::array_t<int> py_cols(h_cols.extent(0), h_cols.data());
    pybind11::array_t<int> py_rows(h_rows.extent(0), h_rows.data());
    pybind11::array_t<double> py_rhs(h_rhs.extent(0), h_rhs.data());
    return pybind11::make_tuple(py_vals, py_cols, py_rows, py_rhs);
  }

  // --- GLOBAL SYSTEM MEMORY ---
  ViewCSRVals d_global_vals;
  ViewCSRCols d_global_cols;
  ViewCSRRows d_global_rows;
  ViewReal1D d_rhs;

  // --- KOKKOS KERNEL LAUNCHER ---
  void assemble(const AggMesh &mesh) {
    Kokkos::deep_copy(d_global_vals, 0.0);
    Kokkos::deep_copy(d_rhs, 0.0);

    AssembleVolumeKernel vol_kernel(
        d_global_vals, d_global_cols, d_global_rows, d_rhs, mesh.d_t_offsets,
        mesh.d_triangles, mesh.d_nodes, mesh.d_bboxes, // Mesh data
        d_orders, d_offsets,                           // dG data
        d_source_nodal,                                // Source field
        d_vx_quad, d_vy_quad,                          // Velocity fields
        d_Kxx_quad, d_Kxy_quad, d_Kyx_quad, d_Kyy_quad // Diffusion
    );

    // 1 thread per polygonal element (mesh.n_elems).
    Kokkos::parallel_for("AssembleVolume", mesh.num_elements(), vol_kernel);
    Kokkos::fence();

    int num_int_faces = mesh.h_faces.size() / 4;

    AssembleFaceKernel face_kernel(
        d_global_vals, d_global_cols, d_global_rows, mesh.d_faces, mesh.d_nodes,
        mesh.d_bboxes, d_orders, d_offsets, d_vx_face, d_vy_face, d_Kxx_face,
        d_Kxy_face, d_Kyx_face, d_Kyy_face, coeffs.alpha);

    Kokkos::parallel_for("AssembleInternalFaces", num_int_faces, face_kernel);
    Kokkos::fence();

    // Boundary Faces
    int num_bnd_faces = mesh.h_bnd_faces.size() / 3;
    AssembleBoundaryKernel bnd_kernel(
        d_global_vals, d_global_cols, d_global_rows, d_rhs, mesh.d_bnd_faces,
        mesh.d_nodes, mesh.d_bboxes, d_orders, d_offsets, d_vx_face, d_vy_face,
        d_Kxx_face, d_Kyy_face, d_g_D_face, d_g_N_face, d_bctype_face,
        coeffs.alpha, num_int_faces);
    Kokkos::parallel_for("AssembleBoundaryFaces", num_bnd_faces, bnd_kernel);
    Kokkos::fence();
  }

  void create_sparse_graph(const AggMesh &mesh) {
    if (total_dofs == 0) {
      throw std::runtime_error(
          "You must call set_p_orders() before!");
    }

    int n_elem = mesh.num_elements();

    // Build an element-to-neighbor adjacency list using the mesh faces
    std::vector<std::vector<int>> elem_neighbors(n_elem);
    int num_faces = mesh.h_faces.size() / 4;
    for (int f = 0; f < num_faces; ++f) {
      int ePlus = mesh.h_faces[4 * f + 0];
      int eMinus = mesh.h_faces[4 * f + 1];
      // Connect the two elements sharing this face
      elem_neighbors[ePlus].push_back(eMinus);
      elem_neighbors[eMinus].push_back(ePlus);
    }

    // Fetch the DOF offsets from the DEVICE back to the CPU
    auto h_offsets =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_offsets);

    for (int e = 0; e < n_elem; ++e) {
      std::sort(elem_neighbors[e].begin(), elem_neighbors[e].end());
      elem_neighbors[e].erase(
          std::unique(elem_neighbors[e].begin(), elem_neighbors[e].end()),
          elem_neighbors[e].end());
    }
    // Build the Sparsity Pattern (List of non-zero columns for each global row)
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

    // Flatten the graph into the CSR 1D arrays
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

    // Allocate the Kokkos DEVICE Views!
    d_global_rows = ViewCSRRows("d_global_rows", total_dofs + 1);
    d_global_cols = ViewCSRCols("d_global_cols", nnz);
    d_global_vals = ViewCSRVals("d_global_vals", nnz); 
    d_rhs = ViewReal1D("d_rhs", total_dofs);           

    // Deep copy the graph structure to the GPU
    auto mirror_rows = Kokkos::create_mirror_view(d_global_rows);
    auto mirror_cols = Kokkos::create_mirror_view(d_global_cols);

    for (int i = 0; i <= total_dofs; ++i)
      mirror_rows(i) = h_rows[i];
    for (int i = 0; i < nnz; ++i)
      mirror_cols(i) = h_cols[i];

    Kokkos::deep_copy(d_global_rows, mirror_rows);
    Kokkos::deep_copy(d_global_cols, mirror_cols);

  }


  void print_matrix(int print_limit = 20) {
    if (total_dofs == 0 || d_global_rows.extent(0) == 0) {
      std::cout << "Matrix is empty or not allocated yet!\n";
      return;
    }

    // Safely pull the CSR arrays from the GPU back to the CPU
    auto h_rows =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_global_rows);
    auto h_cols =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_global_cols);
    auto h_vals =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_global_vals);
    auto h_offs =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_offsets);

    // Call your print function using the Host Views
    print_csr_matrix_blocks(h_rows, h_cols, h_vals, h_offs, print_limit);
  }

  // --- KOKKOS GPU CONJUGATE GRADIENT SOLVER (SYMMETRIC ONLY) ---
  pybind11::array_t<Real> solve_cg(int max_iters = 10000,
                                   Real tolerance = 1e-8) {
    using CrsMatrixType =
        KokkosSparse::CrsMatrix<Real, int, ExecutionSpace, void, int>;
    int nnz = d_global_vals.extent(0);
    CrsMatrixType A("LHS_Matrix", total_dofs, total_dofs, nnz, d_global_vals,
                    d_global_rows, d_global_cols);

    ViewReal1D d_x("x", total_dofs);
    ViewReal1D d_r("r", total_dofs);
    ViewReal1D d_p("p", total_dofs);
    ViewReal1D d_Ap("Ap", total_dofs);

    Kokkos::deep_copy(d_x, 0.0);
    Kokkos::deep_copy(d_r, d_rhs);
    Kokkos::deep_copy(d_p, d_r);

    Real r_dot_r = KokkosBlas::dot(d_r, d_r);
    Real initial_norm = Kokkos::sqrt(r_dot_r);

    if (initial_norm > 1e-14) {
      int iter = 0;
      for (; iter < max_iters; ++iter) {
        KokkosSparse::spmv("N", 1.0, A, d_p, 0.0, d_Ap);
        Real p_dot_Ap = KokkosBlas::dot(d_p, d_Ap);
        Real alpha = r_dot_r / p_dot_Ap;

        KokkosBlas::axpy(alpha, d_p, d_x);
        KokkosBlas::axpy(-alpha, d_Ap, d_r);

        Real r_new_dot_r_new = KokkosBlas::dot(d_r, d_r);
        Real current_norm = Kokkos::sqrt(r_new_dot_r_new);

        if (current_norm / initial_norm < tolerance) {
          std::cout << "   -> CG Converged in " << iter + 1
                    << " iterations! Norm: " << current_norm << std::endl;
          break;
        }

        Real beta = r_new_dot_r_new / r_dot_r;
        KokkosBlas::axpby(1.0, d_r, beta, d_p);
        r_dot_r = r_new_dot_r_new;
      }
      if (iter == max_iters) {
        std::cout << "   -> Warning: CG did not converge within " << max_iters
                  << " iterations." << std::endl;
      }
    }

    pybind11::array_t<Real> py_x(total_dofs);
    Kokkos::View<Real *, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        h_py_x(py_x.mutable_data(), py_x.shape(0));
    Kokkos::deep_copy(h_py_x, d_x);

    return py_x;
  }

  // --- KOKKOS GPU BiCGStab SOLVER (NON-SYMMETRIC MATRICES) ---
  pybind11::array_t<Real> solve_bicgstab(int max_iters = 10000,
                                         Real tolerance = 1e-8) {
    using CrsMatrixType =
        KokkosSparse::CrsMatrix<Real, int, ExecutionSpace, void, int>;
    int nnz = d_global_vals.extent(0);
    CrsMatrixType A("LHS_Matrix", total_dofs, total_dofs, nnz, d_global_vals,
                    d_global_rows, d_global_cols);

    ViewReal1D d_x("x", total_dofs);
    ViewReal1D d_r("r", total_dofs);
    ViewReal1D d_r0_star("r0_star", total_dofs);
    ViewReal1D d_p("p", total_dofs);
    ViewReal1D d_v("v", total_dofs);
    ViewReal1D d_s("s", total_dofs);
    ViewReal1D d_t("t", total_dofs);

    Kokkos::deep_copy(d_x, 0.0);
    Kokkos::deep_copy(d_r, d_rhs);
    Kokkos::deep_copy(d_r0_star, d_r);
    Kokkos::deep_copy(d_p, d_r);

    Real initial_norm = Kokkos::sqrt(KokkosBlas::dot(d_r, d_r));
    Real rho_prev = 1.0, alpha = 1.0, omega = 1.0;

    if (initial_norm > 1e-14) {
      int iter = 0;
      for (; iter < max_iters; ++iter) {
        Real rho = KokkosBlas::dot(d_r0_star, d_r);
        if (std::abs(rho) < 1e-14)
          break;

        if (iter > 0) {
          Real beta = (rho / rho_prev) * (alpha / omega);
          KokkosBlas::axpy(-omega, d_v, d_p);
          KokkosBlas::axpby(1.0, d_r, beta, d_p);
        }

        KokkosSparse::spmv("N", 1.0, A, d_p, 0.0, d_v);
        Real r0_dot_v = KokkosBlas::dot(d_r0_star, d_v);
        if (std::abs(r0_dot_v) < 1e-14)
          break;

        alpha = rho / r0_dot_v;

        Kokkos::deep_copy(d_s, d_r);
        KokkosBlas::axpy(-alpha, d_v, d_s);

        Real norm_s = Kokkos::sqrt(KokkosBlas::dot(d_s, d_s));
        if (norm_s / initial_norm < tolerance) {
          KokkosBlas::axpy(alpha, d_p, d_x);
          std::cout << "   -> BiCGStab Converged in " << iter + 1
                    << " iterations! Norm: " << norm_s << std::endl;
          break;
        }

        KokkosSparse::spmv("N", 1.0, A, d_s, 0.0, d_t);
        Real t_dot_s = KokkosBlas::dot(d_t, d_s);
        Real t_dot_t = KokkosBlas::dot(d_t, d_t);
        omega = t_dot_s / t_dot_t;

        KokkosBlas::axpy(alpha, d_p, d_x);
        KokkosBlas::axpy(omega, d_s, d_x);

        Kokkos::deep_copy(d_r, d_s);
        KokkosBlas::axpy(-omega, d_t, d_r);

        Real current_norm = Kokkos::sqrt(KokkosBlas::dot(d_r, d_r));
        if (current_norm / initial_norm < tolerance) {
          std::cout << "   -> BiCGStab Converged in " << iter + 1
                    << " iterations! Norm: " << current_norm << std::endl;
          break;
        }

        rho_prev = rho;
      }
      if (iter == max_iters) {
        std::cout << "   -> Warning: BiCGStab did not converge within "
                  << max_iters << " iterations." << std::endl;
      }
    }

    pybind11::array_t<Real> py_x(total_dofs);
    Kokkos::View<Real *, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        h_py_x(py_x.mutable_data(), py_x.shape(0));
    Kokkos::deep_copy(h_py_x, d_x);

    return py_x;
  }
};
