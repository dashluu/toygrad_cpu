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
        ADD_ASSIGN, SUB_ASSIGN, MUL_ASSIGN, DIV_ASSIGN, ALIAS, DIFF_ALIAS, PERM,
        EQ, NEQ, LESS, GREATER, LEQ, GEQ, MAX, MIN,
        RELU, SUM, SIGMOID, SOFTMAX,
        COPY
    };

    inline std::unordered_map<OpName, std::string> op2Str = {
        {OpName::INDEX, "INDEX"}, {OpName::CONST, "CONST"}, {OpName::ARANGE, "ARANGE"},
        {OpName::FROM_ARR, "FROM_ARR"}, {OpName::RANDINT, "RANDINT"}, {OpName::RANDN, "RANDN"},
        {OpName::ADD, "ADD"}, {OpName::SUB, "SUB"}, {OpName::MUL, "MUL"}, {OpName::DIV, "DIV"},
        {OpName::POW, "POW"}, {OpName::LOG, "LOG"}, {OpName::SIN, "SIN"}, {OpName::COS, "COS"},
        {OpName::EXP, "EXP"}, {OpName::RECIP, "RECIP"}, {OpName::NEG, "NEG"}, {OpName::SQ, "SQ"},
        {OpName::SQRT, "SQRT"}, {OpName::MATMUL, "MATMUL"}, {OpName::ADD_ASSIGN, "ADD_ASSIGN"},
        {OpName::SUB_ASSIGN, "SUB_ASSIGN"}, {OpName::MUL_ASSIGN, "MUL_ASSIGN"}, {OpName::DIV_ASSIGN, "DIV_ASSIGN"},
        {OpName::ALIAS, "ALIAS"}, {OpName::DIFF_ALIAS, "DIFF_ALIAS"}, {OpName::PERM, "PERM"},
        {OpName::EQ, "EQ"}, {OpName::NEQ, "NEQ"}, {OpName::LESS, "LESS"}, {OpName::GREATER, "GREATER"},
        {OpName::LEQ, "LEQ"}, {OpName::GEQ, "GEQ"}, {OpName::MAX, "MAX"}, {OpName::MIN, "MIN"},
        {OpName::RELU, "RELU"}, {OpName::SUM, "SUM"}, {OpName::SIGMOID, "SIGMOID"}, {OpName::SOFTMAX, "SOFTMAX"},
        {OpName::COPY, "COPY"}
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

    struct LeafOp : Op {
        LeafOp(OpName opName, Tensor *tensor, bool lazy): Op(OpType::LEAF, opName, tensor) {
            if (lazy) {
                tensor->ops.push_back(this);
            }
        }
    };

    struct UnOp : Op {
        TensorPtr operand;

        UnOp(OpName opName, const TensorPtr &operand, Tensor *tensor, bool lazy): Op(OpType::UN_OP, opName, tensor),
            operand(operand) {
            if (lazy) {
                tensor->ops.push_back(this);
                operand->edges.push_back(tensor);
            }
        }
    };

    struct BinOp : Op {
        TensorPtr lhs;
        TensorPtr rhs;

        BinOp(OpName opName, const TensorPtr &lhs, const TensorPtr &rhs, Tensor *tensor, bool lazy): Op(OpType::BIN_OP,
                opName, tensor), lhs(lhs), rhs(rhs) {
            if (lazy) {
                tensor->ops.push_back(this);
                lhs->edges.push_back(tensor);
                rhs->edges.push_back(tensor);
            }
        }
    };

    struct IndexOp final : UnOp {
        std::vector<size_t> idx;

        IndexOp(const TensorPtr &operand, Tensor *tensor, const std::vector<size_t> &idx,
                bool lazy): UnOp(OpName::INDEX, operand, tensor, lazy), idx(idx) {
        }
    };

    struct ConstOp final : LeafOp {
        real c;

        ConstOp(Tensor *tensor, real c, bool lazy) : LeafOp(OpName::CONST, tensor, lazy), c(c) {
        }

        void forward() override;
    };

    struct ArangeOp final : LeafOp {
        real start;
        real step;

        ArangeOp(Tensor *tensor, real start, real step, bool lazy): LeafOp(OpName::ARANGE, tensor, lazy), start(start),
                                                                    step(step) {
        }

        void forward() override;
    };

    struct RandintOp final : LeafOp {
        int64_t min;
        int64_t max;

        RandintOp(Tensor *tensor, int64_t min, int64_t max, bool lazy) : LeafOp(OpName::RANDINT, tensor, lazy),
                                                                         min(min), max(max) {
        }

        void forward() override;
    };

    struct RandnOp final : LeafOp {
        explicit RandnOp(Tensor *tensor, bool lazy) : LeafOp(OpName::RANDN, tensor, lazy) {
        }

        void forward() override;
    };

    struct FromArrOp final : LeafOp {
        const real *data;

        FromArrOp(Tensor *tensor, const real *data, bool lazy) : LeafOp(OpName::FROM_ARR, tensor, lazy), data(data) {
        }

        void forward() override;
    };

    struct SumOp final : UnOp {
        int64_t dim;

        SumOp(const TensorPtr &operand, Tensor *tensor, int64_t dim, bool lazy): UnOp(OpName::SUM, operand, tensor,
                lazy), dim(dim) {
        }

        void forward() override;

        void backward() override;
    };

    struct AddOp final : BinOp {
        AddOp(const TensorPtr &lhs, const TensorPtr &rhs, Tensor *tensor, bool lazy): BinOp(
            OpName::ADD, lhs, rhs, tensor, lazy) {
        }

        void forward() override;

        void backward() override;
    };

    struct AddAssignOp final : UnOp {
        AddAssignOp(const TensorPtr &operand, Tensor *tensor, bool lazy) : UnOp(
            OpName::ADD_ASSIGN, operand, tensor, lazy) {
        }

        void forward() override;
    };

    struct SubOp final : BinOp {
        SubOp(const TensorPtr &lhs, const TensorPtr &rhs, Tensor *tensor, bool lazy): BinOp(
            OpName::SUB, lhs, rhs, tensor, lazy) {
        }

        void forward() override;

        void backward() override;
    };

    struct SubAssignOp final : UnOp {
        SubAssignOp(const TensorPtr &operand, Tensor *tensor, bool lazy) : UnOp(
            OpName::SUB_ASSIGN, operand, tensor, lazy) {
        }

        void forward() override;
    };

    struct MulOp final : BinOp {
        MulOp(const TensorPtr &lhs, const TensorPtr &rhs, Tensor *tensor, bool lazy): BinOp(
            OpName::MUL, lhs, rhs, tensor, lazy) {
        }

        void forward() override;

        void backward() override;
    };

    struct MulAssignOp final : UnOp {
        MulAssignOp(const TensorPtr &operand, Tensor *tensor, bool lazy) : UnOp(
            OpName::MUL_ASSIGN, operand, tensor, lazy) {
        }

        void forward() override;
    };

    struct DivOp final : BinOp {
        DivOp(const TensorPtr &lhs, const TensorPtr &rhs, Tensor *tensor, bool lazy): BinOp(
            OpName::DIV, lhs, rhs, tensor, lazy) {
        }

        void forward() override;

        void backward() override;
    };

    struct DivAssignOp final : UnOp {
        DivAssignOp(const TensorPtr &operand, Tensor *tensor, bool lazy) : UnOp(
            OpName::DIV_ASSIGN, operand, tensor, lazy) {
        }

        void forward() override;
    };

    struct PowOp final : UnOp {
        real c;

        PowOp(const TensorPtr &operand, Tensor *tensor, real c, bool lazy): UnOp(OpName::POW, operand, tensor, lazy),
                                                                            c(c) {
        }

        void forward() override;

        void backward() override;
    };

    struct LogOp final : UnOp {
        LogOp(const TensorPtr &operand, Tensor *tensor, bool lazy): UnOp(OpName::LOG, operand, tensor, lazy) {
        }

        void forward() override;

        void backward() override;
    };

    struct SinOp final : UnOp {
        SinOp(const TensorPtr &operand, Tensor *tensor, bool lazy): UnOp(OpName::SIN, operand, tensor, lazy) {
        }

        void forward() override;

        void backward() override;
    };

    struct CosOp final : UnOp {
        CosOp(const TensorPtr &operand, Tensor *tensor, bool lazy): UnOp(OpName::COS, operand, tensor, lazy) {
        }

        void forward() override;

        void backward() override;
    };

    struct ExpOp final : UnOp {
        ExpOp(const TensorPtr &operand, Tensor *tensor, bool lazy): UnOp(OpName::EXP, operand, tensor, lazy) {
        }

        void forward() override;

        void backward() override;
    };

    struct RecipOp final : UnOp {
        real c;

        RecipOp(const TensorPtr &operand, Tensor *tensor, real c, bool lazy): UnOp(OpName::RECIP, operand, tensor,
                                                                                  lazy), c(c) {
        }

        void forward() override;

        void backward() override;
    };

    struct NegOp final : UnOp {
        NegOp(const TensorPtr &operand, Tensor *tensor, bool lazy): UnOp(OpName::NEG, operand, tensor, lazy) {
        }

        void forward() override;

        void backward() override;
    };

    struct SqOp final : UnOp {
        SqOp(const TensorPtr &operand, Tensor *tensor, bool lazy): UnOp(OpName::SQ, operand, tensor, lazy) {
        }

        void forward() override;

        void backward() override;
    };

    struct SqrtOp final : UnOp {
        SqrtOp(const TensorPtr &operand, Tensor *tensor, bool lazy): UnOp(OpName::SQRT, operand, tensor, lazy) {
        }

        void forward() override;

        void backward() override;
    };

    struct AliasOp final : UnOp {
        AliasOp(const TensorPtr &operand, Tensor *tensor, bool lazy): UnOp(OpName::ALIAS, operand, tensor, lazy) {
        }

        void forward() override;
    };

    struct DiffAliasOp final : UnOp {
        DiffAliasOp(const TensorPtr &operand, Tensor *tensor, bool lazy): UnOp(
            OpName::DIFF_ALIAS, operand, tensor, lazy) {
        }

        void forward() override;

        void backward() override;
    };

    struct EqOp final : BinOp {
        EqOp(const TensorPtr &lhs, const TensorPtr &rhs, Tensor *tensor, bool lazy): BinOp(
            OpName::EQ, lhs, rhs, tensor, lazy) {
        }

        void forward() override;
    };

    struct NeqOp final : BinOp {
        NeqOp(const TensorPtr &lhs, const TensorPtr &rhs, Tensor *tensor, bool lazy): BinOp(
            OpName::NEQ, lhs, rhs, tensor, lazy) {
        }

        void forward() override;
    };

    struct LessOp final : BinOp {
        LessOp(const TensorPtr &lhs, const TensorPtr &rhs, Tensor *tensor, bool lazy): BinOp(
            OpName::LESS, lhs, rhs, tensor, lazy) {
        }

        void forward() override;
    };

    struct GreaterOp final : BinOp {
        GreaterOp(const TensorPtr &lhs, const TensorPtr &rhs, Tensor *tensor, bool lazy): BinOp(
            OpName::GREATER, lhs, rhs, tensor, lazy) {
        }

        void forward() override;
    };

    struct LeqOp final : BinOp {
        LeqOp(const TensorPtr &lhs, const TensorPtr &rhs, Tensor *tensor, bool lazy): BinOp(
            OpName::LEQ, lhs, rhs, tensor, lazy) {
        }

        void forward() override;
    };

    struct GeqOp final : BinOp {
        GeqOp(const TensorPtr &lhs, const TensorPtr &rhs, Tensor *tensor, bool lazy): BinOp(
            OpName::GEQ, lhs, rhs, tensor, lazy) {
        }

        void forward() override;
    };

    struct MaxOp final : UnOp {
        int64_t dim;

        MaxOp(const TensorPtr &operand, Tensor *tensor, int64_t dim, bool lazy): UnOp(OpName::MAX, operand, tensor,
                lazy), dim(dim) {
        }

        void forward() override;

        void backward() override;
    };

    struct MinOp final : UnOp {
        int64_t dim;

        MinOp(const TensorPtr &operand, Tensor *tensor, int64_t dim, bool lazy): UnOp(OpName::MIN, operand, tensor,
                lazy), dim(dim) {
        }

        void forward() override;

        void backward() override;
    };

    struct PermOp final : UnOp {
        PermOp(const TensorPtr &operand, Tensor *tensor, bool lazy): UnOp(OpName::PERM, operand, tensor, lazy) {
        }

        void forward() override;

        void backward() override;
    };

    struct ReluOp final : UnOp {
        ReluOp(const TensorPtr &operand, Tensor *tensor, bool lazy): UnOp(OpName::RELU, operand, tensor, lazy) {
        }

        void forward() override;

        void backward() override;
    };

    struct SigmoidOp final : UnOp {
        SigmoidOp(const TensorPtr &operand, Tensor *tensor, bool lazy): UnOp(OpName::SIGMOID, operand, tensor, lazy) {
        }

        void forward() override;

        void backward() override;
    };

    struct CopyOp final : UnOp {
        CopyOp(const TensorPtr &operand, Tensor *tensor, bool lazy): UnOp(OpName::COPY, operand, tensor, lazy) {
        }

        void forward() override;
    };

    struct MatmulOp final : BinOp {
        TensorPtr lhsTranspose = nullptr;

        MatmulOp(const TensorPtr &lhs, const TensorPtr &rhs, Tensor *tensor, bool lazy): BinOp(
            OpName::MATMUL, lhs, rhs, tensor, lazy) {
        }

        void forward() override;

        void backward() override;
    };
}
