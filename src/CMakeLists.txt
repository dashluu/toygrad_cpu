project(toygrad_cpu_lib)

set(HEADER FILES
        shape.h
        str_assert.h
        tensor.h
        vec.h
        common.h
        ops.h
        rand_gen.h
        tensor_iter.h
        tensor_graph.h
        nn.h
)

set(SRC_FILES
        str_assert.cpp
        tensor.cpp
        vec.cpp
        ops.cpp
        tensor_iter.cpp
        tensor_graph.cpp
        nn.cpp
)

add_library(toygrad_cpu_lib STATIC ${SRC_FILES} ${HEADER_FILES})