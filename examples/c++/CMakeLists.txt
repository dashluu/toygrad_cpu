project(c++_example)

set(GRAPHVIZ "Your GraphViz path")
include_directories("${GRAPHVIZ}/include")
link_directories("${GRAPHVIZ}/lib" "Your path to generated libtoygrad_cpu_lib static library")
add_executable(c++_example main.cpp)
target_link_libraries(c++_example PRIVATE gvc cgraph toygrad_cpu_lib)