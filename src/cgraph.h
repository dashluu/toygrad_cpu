//
// Created by Trung Luu on 7/26/24.
//

#pragma once

#include <tensor.h>

// Computational graph
namespace Toygrad::Tensor {
    class CGraph {
    public:
        void backprop(const TensorPtr& root);
    };
}
