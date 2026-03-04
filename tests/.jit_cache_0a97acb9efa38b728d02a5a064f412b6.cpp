
        #include <Kokkos_Core.hpp>
        #include <cmath>

        extern "C" {
            void eval(double* d_out, const double* d_x, const double* d_y, int n, double t) {
                
                Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space, 
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>> out_view(d_out, n);
                Kokkos::View<const double*, Kokkos::DefaultExecutionSpace::memory_space, 
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>> x_view(d_x, n);
                Kokkos::View<const double*, Kokkos::DefaultExecutionSpace::memory_space, 
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>> y_view(d_y, n);

                Kokkos::parallel_for("JIT_Evaluate", n, KOKKOS_LAMBDA(const int i) {
                    double x = x_view(i);
                    double y = y_view(i);
                    
                    // User's injected math expression
                    out_view(i) = 10.0 * x + y; 
                });
                Kokkos::fence();
            }
        }
        