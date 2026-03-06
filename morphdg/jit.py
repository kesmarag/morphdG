import os
import subprocess
import ctypes
import hashlib

class JITEvaluator:
    def __init__(self, math_expr, func_name="eval"):
        self.math_expr = math_expr
        self.func_name = func_name
        self.so_file = self._generate_and_compile()
        
        # Load the compiled shared library into the current Python process
        self.lib = ctypes.CDLL(os.path.abspath(self.so_file))
        
        # Map the C++ function arguments: (double* out, const double* x, const double* y, int n, double t)
        getattr(self.lib, self.func_name).argtypes = [
            ctypes.c_void_p,  # d_out (device pointer)
            ctypes.c_void_p,  # d_x (device pointer)
            ctypes.c_void_p,  # d_y (device pointer)
            ctypes.c_int,     # n (number of quadrature points)
            ctypes.c_double   # t (current time)
        ]

    def _generate_and_compile(self):
        # Create a unique hash for the expression to cache the compiled library

        cache_dir = "/dev/shm/kokkos_jit_cache" 
        os.makedirs(cache_dir, exist_ok=True)
        
        expr_hash = hashlib.md5(self.math_expr.encode()).hexdigest()
        
        cpp_filename = os.path.join(cache_dir, f"jit_{expr_hash}.cpp")
        so_filename  = os.path.join(cache_dir, f"jit_{expr_hash}.so")

        if os.path.exists(so_filename):
            return so_filename

        # The C++ Kokkos boilerplate
        cpp_code = f"""
        #include <Kokkos_Core.hpp>
        #include <cmath>

        extern "C" {{
            void {self.func_name}(double* d_out, const double* d_x, const double* d_y, int n, double t) {{
                
                Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space, 
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>> out_view(d_out, n);
                Kokkos::View<const double*, Kokkos::DefaultExecutionSpace::memory_space, 
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>> x_view(d_x, n);
                Kokkos::View<const double*, Kokkos::DefaultExecutionSpace::memory_space, 
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>> y_view(d_y, n);

                Kokkos::parallel_for("JIT_Evaluate", n, KOKKOS_LAMBDA(const int i) {{
                    double x = x_view(i);
                    double y = y_view(i);
                    
                    out_view(i) = {self.math_expr}; 
                }});
                Kokkos::fence();
            }}
        }}
        """
        
        with open(cpp_filename, "w") as f:
            f.write(cpp_code)
            
        compile_cmd = [
            "g++", "-O3", "-shared", "-fPIC", 
            "-std=c++20", "-fopenmp",         
            "-I/opt/kokkos/include",           
            cpp_filename, "-o", so_filename, 
            "-L/opt/kokkos/lib64", 
            "-Wl,-rpath=/opt/kokkos/lib64",
            "-lkokkoscore" 
        ]
        
        print(f" -> JIT Compiling Kernel: '{self.math_expr}' ...")
        try:
            subprocess.run(compile_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"JIT Compilation Failed!\n{e.stderr}")
            raise RuntimeError("Failed to compile JIT kernel.")
            
        return so_filename

    def evaluate(self, ptr_out, ptr_x, ptr_y, n_points, current_time):
        """Executes the JIT-compiled Kokkos kernel on the device."""
        func = getattr(self.lib, self.func_name)
        func(ptr_out, ptr_x, ptr_y, n_points, current_time)
