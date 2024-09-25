//
// Created by Trung Luu on 7/16/24.
//

#include "str_assert.h"

namespace Toygrad::Error {
    const std::string Message::gradOnScalarOnly = "Gradient can only be created for scalar output";
    const std::string Message::indexMultidimsOnly =
            "Indexing can only be performed on a tensor with multiple dimensions";
    const std::string Message::indexOutOfBounds = "Indexing element out of bounds";
    const std::string Message::matmulOnLessThan2d =
            "Matrix multiplication can only be applied on tensors with two dimensions or above";
    const std::string Message::invalidShapePerm = "Invalid shape permutation";
    const std::string Message::backpropFromNull = "Cannot backpropagate from a tensor without any gradient";
    const std::string Message::tensorGraphUninitialized =
            "Cannot backpropagate because tensor graph is not initialized";

    std::string Message::invalidDim(int dim, const Shape &shape) {
        return "Invalid dimension " + std::to_string(dim) + " of shape " + shape.toStr();
    }

    std::string Message::notBroadcastable(const Shape &shape1, const Shape &shape2) {
        return "Tensor of shape " + shape1.toStr() + " is not broadcastable to tensor of shape " + shape2.toStr();
    }

    std::string Message::shapesMismatched(const std::string &opNameStr, const Shape &shape1,
                                                const Shape &shape2) {
        return "Shapes mismatched during " + opNameStr + ": " + shape1.toStr() + " and " + shape2.toStr();
    }

    std::string Message::invalidInputSize(size_t actual, size_t expected) {
        return "Expected input of size " + std::to_string(expected) + " but got " + std::to_string(actual);
    }
}
