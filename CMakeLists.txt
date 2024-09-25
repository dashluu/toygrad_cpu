cmake_minimum_required(VERSION 3.28)
project(toygrad_cpu)

set(CMAKE_CXX_STANDARD 20)

# Python and Pybind11
set(PYTHON "Your Python path")
set(PYBIND "Your PyBind include path")
set(CMAKE_CXX_FLAGS "-undefined dynamic_lookup")
include_directories(${PYTHON})
include_directories(${PYBIND})
find_package(pybind11 REQUIRED)

# Graphviz
set(GRAPHVIZ "Your Graphviz path")
include_directories("${GRAPHVIZ}/include")
link_directories("${GRAPHVIZ}/lib")

include_directories(src)
add_subdirectory(src)
pybind11_add_module(toygrad_cpu src/python/pygrad.cpp)
target_link_libraries(toygrad_cpu PRIVATE gvc cgraph)
target_link_libraries(toygrad_cpu PUBLIC toygrad_cpu_lib)
add_subdirectory(tests)
# Optional
# add_subdirectory(examples/c++)