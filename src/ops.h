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
        ADD, SUB, MUL, DIV, EXP, RECIP, NEG, SQ, SQRT,
        ADD_ASSIGN, SUB_ASSIGN, MUL_ASSIGN, DIV_ASSIGN,
        EQ, NEQ, LESS, GREATER, LEQ, GEQ,
        RELU, SUM, SIGMOID,
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
            // operand is never null
            operand->edges.push_back(tensor);
        }
    };

    struct BinOp : Op {
        TensorPtr lhs;
        TensorPtr rhs;

        BinOp(OpName opName, const TensorPtr &lhs, const TensorPtr &rhs, Tensor *tensor): Op(OpType::BIN_OP, opName,
                tensor), lhs(lhs), rhs(rhs) {
            // lhs or rhs can be null depending on the op
            if (lhs) lhs->edges.push_back(tensor);
            if (rhs) rhs->edges.push_back(tensor);
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
        int min;
        int max;

        RandintOp(Tensor *tensor, int min, int max) : Op(OpType::LEAF, OpName::RANDINT, tensor), min(min), max(max) {
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

    // Mask it as a binary op although it makes sense for Sum to be unary op
    struct SumOp final : BinOp {
        int dim;

        SumOp(const TensorPtr &lhs, const TensorPtr &rhs, Tensor *tensor,
              int dim): BinOp(OpName::SUM, lhs, rhs, tensor), dim(dim) {
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

    struct AddAssignOp final : BinOp {
        AddAssignOp(const TensorPtr &lhs, const TensorPtr &rhs, Tensor *tensor) : BinOp(
            OpName::ADD_ASSIGN, lhs, rhs, tensor) {
        }

        void forward() override;

        void backward() override;
    };

    struct SubOp final : BinOp {
        SubOp(const TensorPtr &lhs, const TensorPtr &rhs, Tensor *tensor): BinOp(OpName::SUB, lhs, rhs, tensor) {
        }

        void forward() override;

        void backward() override;
    };

    struct SubAssignOp final : BinOp {
        SubAssignOp(const TensorPtr &lhs, const TensorPtr &rhs, Tensor *tensor) : BinOp(
            OpName::SUB_ASSIGN, lhs, rhs, tensor) {
        }

        void forward() override;

        void backward() override;
    };

    struct MulOp final : BinOp {
        MulOp(const TensorPtr &lhs, const TensorPtr &rhs, Tensor *tensor): BinOp(OpName::MUL, lhs, rhs, tensor) {
        }

        void forward() override;

        void backward() override;
    };

    struct MulAssignOp final : BinOp {
        MulAssignOp(const TensorPtr &lhs, const TensorPtr &rhs, Tensor *tensor) : BinOp(
            OpName::MUL_ASSIGN, lhs, rhs, tensor) {
        }

        void forward() override;

        void backward() override;
    };

    struct DivOp final : BinOp {
        DivOp(const TensorPtr &lhs, const TensorPtr &rhs, Tensor *tensor): BinOp(OpName::DIV, lhs, rhs, tensor) {
        }

        void forward() override;

        void backward() override;
    };

    struct DivAssignOp final : BinOp {
        DivAssignOp(const TensorPtr &lhs, const TensorPtr &rhs, Tensor *tensor) : BinOp(
            OpName::DIV_ASSIGN, lhs, rhs, tensor) {
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
}
