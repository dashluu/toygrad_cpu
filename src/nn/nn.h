//
// Created by Trung Luu on 9/18/24.
//

#pragma once
#include "tensors/tensor_graph.h"

namespace Toygrad::NN {
    class Module {
        Tensor::GraphPtr graph;
        std::vector<Tensor::TensorPtr> input;

    public:
        virtual ~Module() = default;

        Tensor::TensorPtr init(const std::vector<Tensor::TensorPtr> &x);

        void forward() const {
            graph->sort();
            graph->forward();
        }

        void backward() const {
            graph->backward();
        }

        virtual Tensor::TensorPtr forwardFunc(const std::vector<Tensor::TensorPtr> &x) = 0;
    };
}
