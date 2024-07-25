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
        ADD_ASSIGN, SUB_ASSIGN, MUL_ASSIGN, DIV_ASSIGN,
        ADD, SUB, MUL, DIV, EXP,
        EQ, NEQ, LESS, GREATER, LEQ, GEQ,
        RELU, SUM,
        COPY
    };

    struct Op {
        OpType opType;
        OpName opName;
        Tensor *tensor;
        Tensor *grad = nullptr;
        std::vector<Op *> edges = std::vector<Op *>();

        Op(OpType opType, OpName opName, Tensor *tensor) : opType(opType), opName(opName), tensor(tensor) {
            tensor->op = this;
        }

        virtual ~Op() {
            delete tensor;
            delete grad;
        }

        virtual void forward() {
        }

        virtual void backward() {
        }

        void initGrad() {
            if (grad == nullptr) {
                grad = new Tensor(tensor->shape);
            }
        }
    };

    struct UnOp : Op {
        Op *operand = nullptr;

        UnOp(OpName opName, Op *operand, Tensor *tensor): Op(OpType::UN_OP, opName, tensor), operand(operand) {
            operand->edges.push_back(this);
        }
    };

    struct BinOp : Op {
        Op *lhs = nullptr;
        Op *rhs = nullptr;

        BinOp(OpName opName, Op *lhs, Op *rhs, Tensor *tensor): Op(OpType::BIN_OP, opName, tensor), lhs(lhs),
                                                                rhs(rhs) {
            lhs->edges.push_back(this);
            rhs->edges.push_back(this);
        }
    };

    struct IndexOp final : UnOp {
        std::vector<size_t> idx;

        explicit IndexOp(Op *operand, Tensor *tensor, const std::vector<size_t> &idx): UnOp(OpName::INDEX, operand,
                tensor), idx(idx) {
        }
    };

    struct ConstOp final : Op {
        real c;

        explicit ConstOp(Tensor *tensor, real c) : Op(OpType::LEAF, OpName::CONST, tensor), c(c) {
        }
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

        RandintOp(Tensor *tensor, int min, int max) : Op(OpType::LEAF, OpName::RANDINT, tensor), min(min),
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
        SumOp(Op *operand, Tensor *tensor): UnOp(OpName::SUM, operand, tensor) {
        }

        void forward() override;

        void backward() override;
    };

    struct AddOp final : BinOp {
        AddOp(Op *lhs, Op *rhs, Tensor *tensor): BinOp(OpName::ADD, lhs, rhs, tensor) {
        }

        void forward() override;

        void backward() override;
    };

    struct AddAssignOp final : UnOp {
        AddAssignOp(Op *operand, Tensor *tensor): UnOp(OpName::ADD_ASSIGN, operand, tensor) {
        }

        void forward() override;
    };

    struct SubOp final : BinOp {
        SubOp(Op *lhs, Op *rhs, Tensor *tensor): BinOp(OpName::SUB, lhs, rhs, tensor) {
        }

        void forward() override;

        void backward() override;
    };

    struct SubAssignOp final : UnOp {
        SubAssignOp(Op *operand, Tensor *tensor): UnOp(OpName::SUB_ASSIGN, operand, tensor) {
        }

        void forward() override;
    };

    struct MulOp final : BinOp {
        MulOp(Op *lhs, Op *rhs, Tensor *tensor): BinOp(OpName::MUL, lhs, rhs, tensor) {
        }

        void forward() override;

        void backward() override;
    };

    struct MulAssignOp final : UnOp {
        MulAssignOp(Op *operand, Tensor *tensor): UnOp(OpName::MUL_ASSIGN, operand, tensor) {
        }

        void forward() override;
    };

    struct DivOp final : BinOp {
        DivOp(Op *lhs, Op *rhs, Tensor *tensor): BinOp(OpName::DIV, lhs, rhs, tensor) {
        }

        void forward() override;

        void backward() override;
    };

    struct DivAssignOp final : UnOp {
        DivAssignOp(Op *operand, Tensor *tensor): UnOp(OpName::DIV_ASSIGN, operand, tensor) {
        }

        void forward() override;
    };

    struct ExpOp final : UnOp {
        ExpOp(Op *operand, Tensor *tensor): UnOp(OpName::EXP, operand, tensor) {
        }

        void forward() override;

        void backward() override;
    };

    struct EqOp final : BinOp {
        EqOp(Op *lhs, Op *rhs, Tensor *tensor): BinOp(OpName::EQ, lhs, rhs, tensor) {
        }

        void forward() override;
    };

    struct NeqOp final : BinOp {
        NeqOp(Op *lhs, Op *rhs, Tensor *tensor): BinOp(OpName::NEQ, lhs, rhs, tensor) {
        }

        void forward() override;
    };

    struct LessOp final : BinOp {
        LessOp(Op *lhs, Op *rhs, Tensor *tensor): BinOp(OpName::LESS, lhs, rhs, tensor) {
        }

        void forward() override;
    };

    struct GreaterOp final : BinOp {
        GreaterOp(Op *lhs, Op *rhs, Tensor *tensor): BinOp(OpName::GREATER, lhs, rhs, tensor) {
        }

        void forward() override;
    };

    struct LeqOp final : BinOp {
        LeqOp(Op *lhs, Op *rhs, Tensor *tensor): BinOp(OpName::LEQ, lhs, rhs, tensor) {
        }

        void forward() override;
    };

    struct GeqOp final : BinOp {
        GeqOp(Op *lhs, Op *rhs, Tensor *tensor): BinOp(OpName::GEQ, lhs, rhs, tensor) {
        }

        void forward() override;
    };

    struct ReluOp final : UnOp {
        ReluOp(Op *operand, Tensor *tensor): UnOp(OpName::RELU, operand, tensor) {
        }

        void forward() override;

        void backward() override;
    };

    struct CopyOp final : UnOp {
        CopyOp(Op *operand, Tensor *tensor): UnOp(OpName::COPY, operand, tensor) {
        }

        void forward() override;
    };
}
