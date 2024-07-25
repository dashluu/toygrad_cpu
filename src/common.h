//
// Created by Trung Luu on 7/20/24.
//

#pragma once

namespace Toygrad::Tensor {
    using real = float;

    struct Range {
        size_t beg;
        size_t end;
        size_t step;
    };

    class Tensor;
    class TensorIndexer;
    struct Op;
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
    struct ExpOp;
    struct EqOp;
    struct NeqOp;
    struct LessOp;
    struct GreaterOp;
    struct LeqOp;
    struct GeqOp;
    struct ReluOp;
}
