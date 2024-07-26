project(toygrad_cpu_lib)

set(HEADER FILES
        shape.h
        str_assert.h
        tensor.h
        vec.h
        common.h
        ops.h
        tensor_indexer.h
        rand_gen.h
        tensor_iter.h
)

set(SRC_FILES
        str_assert.cpp
        tensor.cpp
        vec.cpp
        ops.cpp
        tensor_indexer.cpp
        tensor_iter.cpp
)

add_library(toygrad_cpu_lib STATIC ${SRC_FILES} ${HEADER_FILES})