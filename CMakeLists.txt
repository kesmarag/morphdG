cmake_minimum_required(VERSION 3.18)
project(morphdG LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the Heavyweight Math Libraries
find_package(Kokkos REQUIRED)
find_package(KokkosKernels REQUIRED)

# Tell Pybind11 to use the modern Python search method!
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)


# Create the Python Module
# When you add files like aggmesh.cpp or physics.cpp, add them to this list!
pybind11_add_module(morphdg_core 
    bindings/pybind_wrapper.cpp 
)

# Link Kokkos to the Python Module
target_link_libraries(morphdg_core PRIVATE Kokkos::kokkos Kokkos::kokkoskernels)

set_target_properties(morphdg_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/morphdg)
