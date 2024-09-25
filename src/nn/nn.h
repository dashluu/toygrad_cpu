//
// Created by Trung Luu on 9/18/24.
//

#pragma once
#include "tensors/tensor_graph.h"

namespace Toygrad::NN {
    class Module {
        std::vector<Tensor::TensorPtr> input;
        Tensor::TensorPtr output = nullptr;

    public:
        virtual ~Module() = default;

        Tensor::TensorPtr forward(const std::vector<Tensor::TensorPtr> &x);

        virtual Tensor::TensorPtr F(const std::vector<Tensor::TensorPtr> &x) = 0;
    };
}
