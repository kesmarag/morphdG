#include "../include/morphdg/aggmesh.hpp"
#include "../include/morphdg/dgsolver.hpp"
#include <Kokkos_Core.hpp>
#include <pybind11/numpy.h> // py::array_t
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// --- Zero-Copy std::vector to Numpy Array ---
template <typename T>
py::array_t<T> as_pyarray(std::vector<T> &vec, py::object parent) {
  return py::array_t<T>({vec.size()}, // Array size
                        {sizeof(T)},  // Stride in bytes
                        vec.data(),   // Raw memory pointer!
                        parent        // Tie lifetime to the C++ object
  );
}

// RAII Manager to safely start/stop Kokkos
struct KokkosManager {
  KokkosManager() {
    if (!Kokkos::is_initialized()) {
      Kokkos::initialize();
    }
  }
  ~KokkosManager() {
    if (Kokkos::is_initialized()) {
      Kokkos::finalize();
    }
  }
};

PYBIND11_MODULE(morphdg_core, m) {
  py::class_<KokkosManager>(m, "KokkosManager").def(py::init<>());

  py::class_<AggMesh>(m, "AggMesh")
      .def(py::init<>())
      .def("load", &AggMesh::load_base_mesh)
      .def("random_agglomerate", &AggMesh::generate_random_polygons)

      // Properties
      .def_property_readonly("num_nodes", &AggMesh::num_nodes)
      .def_property_readonly("num_elements", &AggMesh::num_elements)
      .def_property_readonly("num_triangles", &AggMesh::num_triangles)

      // RAW MEMORY ACCESS (ZERO-COPY NUMPY ARRAYS)

      .def_property_readonly("nodes",
                             [](py::object &obj) {
                               return as_pyarray(obj.cast<AggMesh &>().h_nodes,
                                                 obj);
                             })
      .def_property_readonly("triangles",
                             [](py::object &obj) {
                               return as_pyarray(
                                   obj.cast<AggMesh &>().h_triangles, obj);
                             })
      .def_property_readonly("faces",
                             [](py::object &obj) {
                               return as_pyarray(obj.cast<AggMesh &>().h_faces,
                                                 obj);
                             })
      .def_property_readonly("bnd_faces",
                             [](py::object &obj) {
                               return as_pyarray(
                                   obj.cast<AggMesh &>().h_bnd_faces, obj);
                             })
      .def_property_readonly("t_offsets",
                             [](py::object &obj) {
                               return as_pyarray(
                                   obj.cast<AggMesh &>().h_t_offsets, obj);
                             })
      .def("push_to_device", &AggMesh::push_to_device);

  // py::class_<Coeffs>(m, "Coeffs")
  // .def(py::init<>())
  // .def_readwrite("Kxx", &Coeffs::Kxx)
  // .def_readwrite("Kxy", &Coeffs::Kxy)
  // .def_readwrite("Kyx", &Coeffs::Kyx)
  // .def_readwrite("Kyy", &Coeffs::Kyy)
  // .def_readwrite("vx", &Coeffs::vx)
  // .def_readwrite("vy", &Coeffs::vy)
  // .def_readwrite("alpha", &Coeffs::alpha);

  py::class_<DGSolver>(m, "DGSolver")
      .def(py::init<>())
      // .def_readwrite("coeffs", &DGSolver::coeffs)
      .def("vol_quad_points", &DGSolver::vol_quad_points, py::arg("mesh"))
      .def_readonly("total_dofs", &DGSolver::total_dofs)
      // The Zero-Copy Lambda Bridge for NumPy Arrays
      .def("set_p_orders",
           [](DGSolver &solver, py::array_t<int> arr) {
             py::buffer_info buf = arr.request();

             // Safety check: Ensure it's a 1D array
             if (buf.ndim != 1) {
               throw std::runtime_error("p_orders must be a 1D NumPy array");
             }

             // Pass the raw memory pointer and size straight to C++
             solver.set_p_orders(static_cast<const int *>(buf.ptr),
                                 buf.shape[0]);
           })

      .def("set_vx_field",
           [](DGSolver &solver, py::array_t<double> arr) {
             py::buffer_info buf = arr.request();
             if (buf.ndim != 1)
               throw std::runtime_error("vx_field must be a 1D NumPy array");
             solver.set_vx_field(static_cast<const double *>(buf.ptr),
                                 buf.shape[0]);
           })

      .def("set_vy_field",
           [](DGSolver &solver, py::array_t<double> arr) {
             py::buffer_info buf = arr.request();
             if (buf.ndim != 1)
               throw std::runtime_error("vy_field must be a 1D NumPy array");
             solver.set_vy_field(static_cast<const double *>(buf.ptr),
                                 buf.shape[0]);
           })

      .def("set_Kxx_field",
           [](DGSolver &s, py::array_t<double> a) {
             py::buffer_info b = a.request();
             s.set_Kxx_field(static_cast<const double *>(b.ptr), b.shape[0]);
           })
      .def("set_Kxy_field",
           [](DGSolver &s, py::array_t<double> a) {
             py::buffer_info b = a.request();
             s.set_Kxy_field(static_cast<const double *>(b.ptr), b.shape[0]);
           })
      .def("set_Kyx_field",
           [](DGSolver &s, py::array_t<double> a) {
             py::buffer_info b = a.request();
             s.set_Kyx_field(static_cast<const double *>(b.ptr), b.shape[0]);
           })
      .def("set_Kyy_field",
           [](DGSolver &s, py::array_t<double> a) {
             py::buffer_info b = a.request();
             s.set_Kyy_field(static_cast<const double *>(b.ptr), b.shape[0]);
           })
      .def("assemble_volume", &DGSolver::assemble_volume, py::arg("mesh"))
      .def("create_sparse_graph", &DGSolver::create_sparse_graph,
           py::arg("mesh"))
      .def("set_source_nodal", [](DGSolver &s, py::array_t<double> a) {
        py::buffer_info b = a.request();
        if (b.ndim != 1)
          throw std::runtime_error("Source field must be a 1D NumPy array");
        s.set_source_nodal(static_cast<const double *>(b.ptr), b.shape[0]);
      }).def("print_matrix", &DGSolver::print_matrix, py::arg("limit") = 20);
}
