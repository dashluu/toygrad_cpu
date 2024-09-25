project(toygrad_cpu_lib)

set(HEADER FILES
        tensors/shape.h
        assert/str_assert.h
        tensors/tensor.h
        tensors/vec.h
        common.h
        tensors/ops.h
        tensors/rand_gen.h
        tensors/tensor_iter.h
        tensors/tensor_graph.h
        nn/nn.h
        nn/linear.h
        tensors/tensor_draw.h
)

set(SRC_FILES
        assert/str_assert.cpp
        tensors/tensor.cpp
        tensors/vec.cpp
        tensors/ops.cpp
        tensors/tensor_iter.cpp
        tensors/tensor_graph.cpp
        nn/nn.cpp
        nn/linear.cpp
        tensors/tensor_draw.cpp
)

add_library(toygrad_cpu_lib STATIC ${SRC_FILES} ${HEADER_FILES})