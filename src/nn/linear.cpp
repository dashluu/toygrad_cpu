//
// Created by Trung Luu on 9/19/24.
//

#include "linear.h"

namespace Toygrad::NN {
    Tensor::TensorPtr Linear::F(const std::vector<Tensor::TensorPtr> &x) {
        return x[0]->matmul(A)->add(b);
    }
}
