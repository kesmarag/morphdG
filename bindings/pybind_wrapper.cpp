#include <Kokkos_Core.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h> // py::array_t
#include "../include/morphdg/aggmesh.hpp"

namespace py = pybind11;


// --- Zero-Copy std::vector to Numpy Array ---
template <typename T>
py::array_t<T> as_pyarray(std::vector<T>& vec, py::object parent) {
    return py::array_t<T>(
        {vec.size()},          // Array size
        {sizeof(T)},           // Stride in bytes
        vec.data(),            // Raw memory pointer!
        parent                 // Tie lifetime to the C++ object
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
        .def("load_base_mesh", &AggMesh::load_base_mesh)
        .def("generate_random_polygons", &AggMesh::generate_random_polygons)
        
        // Properties
        .def_property_readonly("num_nodes", &AggMesh::num_nodes)
        .def_property_readonly("num_elements", &AggMesh::num_elements)
        .def_property_readonly("num_triangles", &AggMesh::num_triangles)

        // ==========================================================
        // RAW MEMORY ACCESS (ZERO-COPY NUMPY ARRAYS)
        // ==========================================================
        .def_property_readonly("nodes", [](py::object& obj) { 
            return as_pyarray(obj.cast<AggMesh&>().h_nodes, obj); 
        })
        .def_property_readonly("triangles", [](py::object& obj) { 
            return as_pyarray(obj.cast<AggMesh&>().h_triangles, obj); 
        })
        .def_property_readonly("faces", [](py::object& obj) { 
            return as_pyarray(obj.cast<AggMesh&>().h_faces, obj); 
        })
        .def_property_readonly("bnd_faces", [](py::object& obj) { 
            return as_pyarray(obj.cast<AggMesh&>().h_bnd_faces, obj); 
        })
        .def_property_readonly("t_offsets", [](py::object& obj) { 
            return as_pyarray(obj.cast<AggMesh&>().h_t_offsets, obj); 
        });
}




