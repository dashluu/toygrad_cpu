//
// Created by Trung Luu on 7/15/24.
//

#pragma once

#include <iostream>
#include <cassert>

namespace Toygrad::Tensor {
    struct AssertMessage {
        static const std::string gradOnScalarOnly;
        static const std::string indexMultidimsOnly;
        static const std::string indexOutOfBounds;
        static const std::string shapesMismatched;
        static const std::string matmulOnLessThan2d;
        static const std::string invalidShapePerm;
        static const std::string invalidDim;
        static const std::string notBroadcastable;
        static const std::string notSqueezable;
        static const std::string backpropFromNull;
    };

    inline bool str_assert(bool assertion, const std::string &message) {
        if (!assertion) {
            std::cout << message << std::endl;
        }

        return assertion;
    }
}
