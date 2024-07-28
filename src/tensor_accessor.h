//
// Created by Trung Luu on 7/20/24.
//

#pragma once

#include "tensor.h"

namespace Toygrad::Tensor {
    class TensorAccessor {
        Tensor *tensor;

    public:
        explicit TensorAccessor(Tensor *tensor): tensor(tensor) {
        }

        TensorPtr at(const std::vector<size_t> &idx);

        TensorPtr at(const std::vector<Range> &ranges);
    };
}
