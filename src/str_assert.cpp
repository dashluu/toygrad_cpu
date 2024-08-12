//
// Created by Trung Luu on 7/16/24.
//

#include "str_assert.h"

namespace Toygrad::Tensor {
    const std::string AssertMessage::gradOnScalarOnly = "Gradient can only be created for scalar output";
    const std::string AssertMessage::indexMultidimsOnly =
            "Indexing can only be performed on a tensor with multiple dimensions";
    const std::string AssertMessage::indexOutOfBounds = "Indexing element out of bounds";
    const std::string AssertMessage::shapesMismatched = "Shapes mismatched";
    const std::string AssertMessage::matmulOnLessThan2d =
            "Matrix multiplication can only be applied on tensors with 2d or above";
    const std::string AssertMessage::invalidShapePerm = "Invalid shape permutation";
    const std::string AssertMessage::invalidDim = "Invalid dimension";
    const std::string AssertMessage::notBroadcastable = "Tensors are not broadcastable";
    const std::string AssertMessage::notSqueezable = "Tensors are not squeezable";
}
