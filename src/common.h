//
// Created by Trung Luu on 7/20/24.
//

#pragma once

#include <cmath>

namespace Toygrad::Tensor {
    using real = float;

    struct Range {
        size_t beg;
        size_t end;
        size_t step;
    };

    class CGraph;
    class TensorIter;
    class Tensor;
    struct Op;
    struct UnOp;
    struct BinOp;
    struct IndexOp;
    struct ConstOp;
    struct ArangeOp;
    struct RandintOp;
    struct RandnOp;
    struct FromArrOp;
    struct SumOp;
    struct AddOp;
    struct AddAssignOp;
    struct SubOp;
    struct SubAssignOp;
    struct MulOp;
    struct MulAssignOp;
    struct DivOp;
    struct DivAssignOp;
    struct PowOp;
    struct LogOp;
    struct SinOp;
    struct CosOp;
    struct ExpOp;
    struct RecipOp;
    struct NegOp;
    struct SqOp;
    struct SqrtOp;
    struct AliasOp;
    struct EqOp;
    struct NeqOp;
    struct LessOp;
    struct GreaterOp;
    struct LeqOp;
    struct GeqOp;
    struct MaxOp;
    struct MinOp;
    struct PermOp;
    struct ReluOp;
    struct SigmoidOp;
    struct SoftmaxOp;
    struct CopyOp;
    struct MatmulOp;

    using TensorPtr = std::shared_ptr<Tensor>;
    using IterPtr = std::unique_ptr<TensorIter>;
    const auto sqrt = std::sqrtf;
    const auto exp = std::expf;
    const auto pow = std::powf;
    const auto sin = std::sinf;
    const auto cos = std::cosf;
    const auto log = std::logf;
}
