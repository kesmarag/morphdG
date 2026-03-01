#pragma once
#include <Kokkos_Core.hpp>
#include <vector>

struct Coeffs {
  // Diffusion Tensor
  double Kxx = 0.0;
  double Kxy = 0.0;
  double Kyx = 0.0;
  double Kyy = 0.0;

  // Advection Velocity
  double vx = 0.0;
  double vy = 0.0;

  // DG Penalty Parameter
  double alpha = 0.0;
};

struct DGSolver {
  Coeffs coeffs;

  Kokkos::View<double *, Kokkos::LayoutLeft,
               Kokkos::DefaultExecutionSpace::memory_space>
      d_source_nodal;
  Kokkos::View<double *, Kokkos::LayoutLeft,
               Kokkos::DefaultExecutionSpace::memory_space>
      d_boundary_nodal;

  DGSolver() = default;
};
