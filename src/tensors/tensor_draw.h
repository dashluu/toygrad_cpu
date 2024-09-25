//
// Created by Trung Luu on 9/19/24.
//

#pragma once
#include "tensors/tensor.h"

namespace Toygrad::Tensor {
    class TensorDraw {
        static char *strToCharPtr(const std::string &str);

    public:
        void draw(Tensor *root, const std::string &extension, const std::string &fileName);
    };
}
