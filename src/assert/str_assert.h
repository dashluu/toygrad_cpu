//
// Created by Trung Luu on 7/15/24.
//

#pragma once

#include <iostream>
#include <cassert>
#include "tensors/shape.h"

namespace Toygrad::Error {
    using Tensor::Shape;

    struct Message {
        static const std::string gradOnScalarOnly;
        static const std::string indexMultidimsOnly;
        static const std::string indexOutOfBounds;
        static const std::string matmulOnLessThan2d;
        static const std::string invalidShapePerm;
        static const std::string backpropFromNull;
        static const std::string tensorGraphUninitialized;

        static std::string invalidDim(int dim, const Shape &shape);

        static std::string notBroadcastable(const Shape &shape1, const Shape &shape2);

        static std::string shapesMismatched(const std::string &opNameStr, const Shape &shape1, const Shape &shape2);

        static std::string invalidInputSize(size_t actual, size_t expected);
    };

    inline bool str_assert(bool assertion, const std::string &message) {
        if (!assertion) {
            std::cout << message << std::endl;
        }

        return assertion;
    }
}
