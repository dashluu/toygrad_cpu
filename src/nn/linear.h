//
// Created by Trung Luu on 9/19/24.
//

#pragma once
#include "nn.h"

namespace Toygrad::NN {
    class Linear : public Module {
        Tensor::TensorPtr A;
        Tensor::TensorPtr b;

    public:
        Linear(size_t inputSize, size_t outputSize) {
            A = Tensor::Tensor::randn({inputSize, outputSize});
            b = Tensor::Tensor::randn({outputSize});
        }

        Tensor::TensorPtr F(const std::vector<Tensor::TensorPtr> &x) override;
    };
}
