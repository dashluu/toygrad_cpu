cmake_minimum_required(VERSION 3.28)
project(toygrad_cpu)

set(CMAKE_CXX_STANDARD 20)

include_directories(src)
add_subdirectory(src)
add_executable(toygrad_cpu main.cpp)
target_link_libraries(toygrad_cpu toygrad_cpu_lib)
