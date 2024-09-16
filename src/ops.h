//
// Created by Trung Luu on 7/12/24.
//

#pragma once

#include "rand_gen.h"
#include "tensor.h"

namespace Toygrad::Tensor {
    enum class OpType {
        LEAF, UN_OP, BIN_OP
    };

    enum class OpName {
        INDEX, CONST, ARANGE, FROM_ARR, RANDINT, RANDN,
        ADD, SUB, MUL, DIV, POW, LOG, SIN, COS, EXP, RECIP, NEG, SQ, SQRT, MATMUL,
        ADD_ASSIGN, SUB_ASSIGN, MUL_ASSIGN, DIV_ASSIGN, ALIAS, PERM,
        EQ, NEQ, LESS, GREATER, LEQ, GEQ, MAX, MIN,
        RELU, SUM, SIGMOID, SOFTMAX,
        COPY
    };

    struct Op {
        OpType opType;
        OpName opName;
        Tensor *tensor;

        Op(OpType opType, OpName opName, Tensor *tensor) : opType(opType), opName(opName), tensor(tensor) {
        }

        virtual ~Op() = default;

        virtual void forward() {
        }

        virtual void backward() {
        }
    };

    struct UnOp : Op {
        TensorPtr operand;

        UnOp(OpName opName, const TensorPtr &operand, Tensor *tensor): Op(OpType::UN_OP, opName, tensor),
                                                                       operand(operand) {
            operand->edges.push_back(tensor);
        }
    };

    struct BinOp : Op {
        TensorPtr lhs;
        TensorPtr rhs;

        BinOp(OpName opName, const TensorPtr &lhs, const TensorPtr &rhs, Tensor *tensor): Op(OpType::BIN_OP, opName,
            tensor), lhs(lhs), rhs(rhs) {
            lhs->edges.push_back(tensor);
            rhs->edges.push_back(tensor);
        }
    };

    struct IndexOp final : UnOp {
        std::vector<size_t> idx;

        IndexOp(const TensorPtr &operand, Tensor *tensor, const std::vector<size_t> &idx): UnOp(OpName::INDEX, operand,
                tensor),
            idx(idx) {
        }
    };

    struct ConstOp final : Op {
        real c;

        ConstOp(Tensor *tensor, real c) : Op(OpType::LEAF, OpName::CONST, tensor), c(c) {
        }

        void forward() override;
    };

    struct ArangeOp final : Op {
        real start;
        real step;

        ArangeOp(Tensor *tensor, real start, real step): Op(OpType::LEAF, OpName::ARANGE, tensor), start(start),
                                                         step(step) {
        }

        void forward() override;
    };

    struct RandintOp final : Op {
        int64_t min;
        int64_t max;

        RandintOp(Tensor *tensor, int64_t min, int64_t max) : Op(OpType::LEAF, OpName::RANDINT, tensor), min(min),
                                                              max(max) {
        }

        void forward() override;
    };

    struct RandnOp final : Op {
        explicit RandnOp(Tensor *tensor) : Op(OpType::LEAF, OpName::RANDN, tensor) {
        }

        void forward() override;
    };

    struct FromArrOp final : Op {
        const real *data;

        FromArrOp(Tensor *tensor, const real *data) : Op(OpType::LEAF, OpName::FROM_ARR, tensor), data(data) {
        }

        void forward() override;
    };

    struct SumOp final : UnOp {
        int64_t dim;

        SumOp(const TensorPtr &operand, Tensor *tensor, int64_t dim): UnOp(OpName::SUM, operand, tensor), dim(dim) {
        }

        void forward() override;

        void backward() override;
    };

    struct AddOp final : BinOp {
        AddOp(const TensorPtr &lhs, const TensorPtr &rhs, Tensor *tensor): BinOp(OpName::ADD, lhs, rhs, tensor) {
        }

        void forward() override;

        void backward() override;
    };

    struct AddAssignOp final : UnOp {
        AddAssignOp(const TensorPtr &operand, Tensor *tensor) : UnOp(OpName::ADD_ASSIGN, operand, tensor) {
        }

        void forward() override;
    };

    struct SubOp final : BinOp {
        SubOp(const TensorPtr &lhs, const TensorPtr &rhs, Tensor *tensor): BinOp(OpName::SUB, lhs, rhs, tensor) {
        }

        void forward() override;

        void backward() override;
    };

    struct SubAssignOp final : UnOp {
        SubAssignOp(const TensorPtr &operand, Tensor *tensor) : UnOp(OpName::SUB_ASSIGN, operand, tensor) {
        }

        void forward() override;
    };

    struct MulOp final : BinOp {
        MulOp(const TensorPtr &lhs, const TensorPtr &rhs, Tensor *tensor): BinOp(OpName::MUL, lhs, rhs, tensor) {
        }

        void forward() override;

        void backward() override;
    };

    struct MulAssignOp final : UnOp {
        MulAssignOp(const TensorPtr &operand, Tensor *tensor) : UnOp(OpName::MUL_ASSIGN, operand, tensor) {
        }

        void forward() override;
    };

    struct DivOp final : BinOp {
        DivOp(const TensorPtr &lhs, const TensorPtr &rhs, Tensor *tensor): BinOp(OpName::DIV, lhs, rhs, tensor) {
        }

        void forward() override;

        void backward() override;
    };

    struct DivAssignOp final : UnOp {
        DivAssignOp(const TensorPtr &operand, Tensor *tensor) : UnOp(OpName::DIV_ASSIGN, operand, tensor) {
        }

        void forward() override;
    };

    struct PowOp final : UnOp {
        real c;

        PowOp(const TensorPtr &operand, Tensor *tensor, real c): UnOp(OpName::POW, operand, tensor), c(c) {
        }

        void forward() override;

        void backward() override;
    };

    struct LogOp final : UnOp {
        LogOp(const TensorPtr &operand, Tensor *tensor): UnOp(OpName::LOG, operand, tensor) {
        }

        void forward() override;

        void backward() override;
    };

    struct SinOp final : UnOp {
        SinOp(const TensorPtr &operand, Tensor *tensor): UnOp(OpName::SIN, operand, tensor) {
        }

        void forward() override;

        void backward() override;
    };

    struct CosOp final : UnOp {
        CosOp(const TensorPtr &operand, Tensor *tensor): UnOp(OpName::COS, operand, tensor) {
        }

        void forward() override;

        void backward() override;
    };

    struct ExpOp final : UnOp {
        ExpOp(const TensorPtr &operand, Tensor *tensor): UnOp(OpName::EXP, operand, tensor) {
        }

        void forward() override;

        void backward() override;
    };

    struct RecipOp final : UnOp {
        real c;

        RecipOp(const TensorPtr &operand, Tensor *tensor, real c): UnOp(OpName::RECIP, operand, tensor), c(c) {
        }

        void forward() override;

        void backward() override;
    };

    struct NegOp final : UnOp {
        explicit NegOp(const TensorPtr &operand, Tensor *tensor): UnOp(OpName::NEG, operand, tensor) {
        }

        void forward() override;

        void backward() override;
    };

    struct SqOp final : UnOp {
        explicit SqOp(const TensorPtr &operand, Tensor *tensor): UnOp(OpName::SQ, operand, tensor) {
        }

        void forward() override;

        void backward() override;
    };

    struct SqrtOp final : UnOp {
        explicit SqrtOp(const TensorPtr &operand, Tensor *tensor): UnOp(OpName::SQRT, operand, tensor) {
        }

        void forward() override;

        void backward() override;
    };

    struct AliasOp final : UnOp {
        explicit AliasOp(const TensorPtr &operand, Tensor *tensor): UnOp(OpName::ALIAS, operand, tensor) {
        }

        void forward() override;
    };

    struct EqOp final : BinOp {
        EqOp(const TensorPtr &lhs, const TensorPtr &rhs, Tensor *tensor): BinOp(OpName::EQ, lhs, rhs, tensor) {
        }

        void forward() override;
    };

    struct NeqOp final : BinOp {
        NeqOp(const TensorPtr &lhs, const TensorPtr &rhs, Tensor *tensor): BinOp(OpName::NEQ, lhs, rhs, tensor) {
        }

        void forward() override;
    };

    struct LessOp final : BinOp {
        LessOp(const TensorPtr &lhs, const TensorPtr &rhs, Tensor *tensor): BinOp(OpName::LESS, lhs, rhs, tensor) {
        }

        void forward() override;
    };

    struct GreaterOp final : BinOp {
        GreaterOp(const TensorPtr &lhs, const TensorPtr &rhs, Tensor *tensor): BinOp(
            OpName::GREATER, lhs, rhs, tensor) {
        }

        void forward() override;
    };

    struct LeqOp final : BinOp {
        LeqOp(const TensorPtr &lhs, const TensorPtr &rhs, Tensor *tensor): BinOp(OpName::LEQ, lhs, rhs, tensor) {
        }

        void forward() override;
    };

    struct GeqOp final : BinOp {
        GeqOp(const TensorPtr &lhs, const TensorPtr &rhs, Tensor *tensor): BinOp(OpName::GEQ, lhs, rhs, tensor) {
        }

        void forward() override;
    };

    struct MaxOp final : UnOp {
        int64_t dim;

        MaxOp(const TensorPtr &operand, Tensor *tensor, int64_t dim): UnOp(OpName::MAX, operand, tensor), dim(dim) {
        }

        void forward() override;

        void backward() override;
    };

    struct MinOp final : UnOp {
        int64_t dim;

        MinOp(const TensorPtr &operand, Tensor *tensor, int64_t dim): UnOp(OpName::MIN, operand, tensor), dim(dim) {
        }

        void forward() override;

        void backward() override;
    };

    struct PermOp final : UnOp {
        PermOp(const TensorPtr &operand, Tensor *tensor): UnOp(OpName::PERM, operand, tensor) {
        }

        void forward() override;

        void backward() override;
    };

    struct ReluOp final : UnOp {
        ReluOp(const TensorPtr &operand, Tensor *tensor): UnOp(OpName::RELU, operand, tensor) {
        }

        void forward() override;

        void backward() override;
    };

    struct SigmoidOp final : UnOp {
        SigmoidOp(const TensorPtr &operand, Tensor *tensor): UnOp(OpName::SIGMOID, operand, tensor) {
        }

        void forward() override;

        void backward() override;
    };

    struct CopyOp final : UnOp {
        CopyOp(const TensorPtr &operand, Tensor *tensor): UnOp(OpName::COPY, operand, tensor) {
        }

        void forward() override;
    };

    struct MatmulOp final : BinOp {
        MatmulOp(const TensorPtr &lhs, const TensorPtr &rhs, Tensor *tensor): BinOp(OpName::MATMUL, lhs, rhs, tensor) {
        }

        void forward() override;

        void backward() override;
    };
}
