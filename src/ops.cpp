//
// Created by Trung Luu on 7/12/24.
//

#include "ops.h"
#include "tensor_iter.h"

namespace Toygrad::Tensor {
    void ArangeOp::forward() {
        DenseIter iter(tensor);
        size_t i = 0;

        for (iter.start(); iter.hasNext(); iter.next()) {
            iter.curr() = start + step * i;
            i++;
        }
    }

    void RandintOp::forward() {
        DenseIter iter(tensor);

        for (iter.start(); iter.hasNext(); iter.next()) {
            iter.curr() = RandGen::randint(min, max);
        }
    }

    void RandnOp::forward() {
        DenseIter iter(tensor);

        for (iter.start(); iter.hasNext(); iter.next()) {
            iter.curr() = RandGen::randn();
        }
    }

    void FromArrOp::forward() {
        DenseIter iter(tensor);
        size_t i = 0;

        for (iter.start(); iter.hasNext(); iter.next()) {
            iter.curr() = data[i];
            i++;
        }
    }

    void AssignOp::forward() {

    }

    void SumOp::forward() {
        std::unique_ptr<Iter> resultIter = initIter(tensor);
        std::unique_ptr<Iter> operandIter = initIter(operand->tensor);
        real sum = 0;

        for (operandIter->start(); operandIter->hasNext(); operandIter->next()) {
            sum += operandIter->curr();
        }

        resultIter->start();
        resultIter->curr() = sum;
    }

    void SumOp::backward() {
        if (operand->opType != OpType::LEAF) {
            operand->initGrad();
            *(operand->grad) += *grad * 1.;
        }
    }

    void AddOp::forward() {
        std::unique_ptr<Iter> resultIter = initIter(tensor);
        std::unique_ptr<Iter> lhsIter = initIter(lhs->tensor);
        std::unique_ptr<Iter> rhsIter = initIter(rhs->tensor);

        for (resultIter->start(), lhsIter->start(), rhsIter->start();
             resultIter->hasNext();
             resultIter->next(), lhsIter->next(), rhsIter->next()) {
            resultIter->curr() = lhsIter->curr() + rhsIter->curr();
        }
    }

    void AddOp::backward() {
        if (lhs->opType != OpType::LEAF) {
            lhs->initGrad();
            *lhs->grad += *grad * 1.;
        }

        if (rhs->opType != OpType::LEAF) {
            rhs->initGrad();
            *rhs->grad += *grad * 1.;
        }
    }

    void AddAssignOp::forward() {
        std::unique_ptr<Iter> resultIter = initIter(tensor);
        std::unique_ptr<Iter> operandIter = initIter(operand->tensor);

        for (resultIter->start(), operandIter->start();
             resultIter->hasNext();
             resultIter->next(), operandIter->next()) {
            resultIter->curr() += operandIter->curr();
        }
    }

    void SubOp::forward() {
        std::unique_ptr<Iter> resultIter = initIter(tensor);
        std::unique_ptr<Iter> lhsIter = initIter(lhs->tensor);
        std::unique_ptr<Iter> rhsIter = initIter(rhs->tensor);

        for (resultIter->start(), lhsIter->start(), rhsIter->start();
             resultIter->hasNext();
             resultIter->next(), lhsIter->next(), rhsIter->next()) {
            resultIter->curr() = lhsIter->curr() - rhsIter->curr();
        }
    }

    void SubOp::backward() {
        if (lhs->opType != OpType::LEAF) {
            lhs->initGrad();
            *lhs->grad += *grad * 1.;
        }

        if (rhs->opType != OpType::LEAF) {
            rhs->initGrad();
            *rhs->grad += *grad * -1.;
        }
    }

    void SubAssignOp::forward() {
        std::unique_ptr<Iter> resultIter = initIter(tensor);
        std::unique_ptr<Iter> operandIter = initIter(operand->tensor);

        for (resultIter->start(), operandIter->start();
             resultIter->hasNext();
             resultIter->next(), operandIter->next()) {
            resultIter->curr() -= operandIter->curr();
        }
    }

    void MulOp::forward() {
        std::unique_ptr<Iter> resultIter = initIter(tensor);
        std::unique_ptr<Iter> lhsIter = initIter(lhs->tensor);
        std::unique_ptr<Iter> rhsIter = initIter(rhs->tensor);

        for (resultIter->start(), lhsIter->start(), rhsIter->start();
             resultIter->hasNext();
             resultIter->next(), lhsIter->next(), rhsIter->next()) {
            resultIter->curr() = lhsIter->curr() * rhsIter->curr();
        }
    }

    void MulOp::backward() {
        if (lhs->opType != OpType::LEAF) {
            lhs->initGrad();
            *lhs->grad += *grad * *rhs->tensor;
        }

        if (rhs->opType != OpType::LEAF) {
            rhs->initGrad();
            *rhs->grad += *grad * *lhs->tensor;
        }
    }

    void MulAssignOp::forward() {
        std::unique_ptr<Iter> resultIter = initIter(tensor);
        std::unique_ptr<Iter> operandIter = initIter(operand->tensor);

        for (resultIter->start(), operandIter->start();
             resultIter->hasNext();
             resultIter->next(), operandIter->next()) {
            resultIter->curr() *= operandIter->curr();
        }
    }

    void DivOp::forward() {
        std::unique_ptr<Iter> resultIter = initIter(tensor);
        std::unique_ptr<Iter> lhsIter = initIter(lhs->tensor);
        std::unique_ptr<Iter> rhsIter = initIter(rhs->tensor);

        for (resultIter->start(), lhsIter->start(), rhsIter->start();
             resultIter->hasNext();
             resultIter->next(), lhsIter->next(), rhsIter->next()) {
            resultIter->curr() = lhsIter->curr() / rhsIter->curr();
        }
    }

    void DivOp::backward() {
        if (lhs->opType != OpType::LEAF) {
            lhs->initGrad();
            *lhs->grad += *grad * (1. / *rhs->tensor);
        }

        if (rhs->opType != OpType::LEAF) {
            rhs->initGrad();
            *rhs->grad += *grad * (*lhs->tensor * (-1. / (*rhs->tensor * *rhs->tensor)));
        }
    }

    void DivAssignOp::forward() {
        std::unique_ptr<Iter> resultIter = initIter(tensor);
        std::unique_ptr<Iter> operandIter = initIter(operand->tensor);

        for (resultIter->start(), operandIter->start();
             resultIter->hasNext();
             resultIter->next(), operandIter->next()) {
            resultIter->curr() /= operandIter->curr();
        }
    }

    void ExpOp::forward() {
        std::unique_ptr<Iter> resultIter = initIter(tensor);
        std::unique_ptr<Iter> operandIter = initIter(operand->tensor);

        for (resultIter->start(), operandIter->start();
             resultIter->hasNext();
             resultIter->next(), operandIter->next()) {
            resultIter->curr() = std::exp(operandIter->curr());
        }
    }

    void ExpOp::backward() {
        if (operand->opType != OpType::LEAF) {
            operand->initGrad();
            *operand->grad += *grad * *operand->tensor * *tensor;
        }
    }

    void EqOp::forward() {
        std::unique_ptr<Iter> resultIter = initIter(tensor);
        std::unique_ptr<Iter> lhsIter = initIter(lhs->tensor);
        std::unique_ptr<Iter> rhsIter = initIter(rhs->tensor);

        for (resultIter->start(), lhsIter->start(), rhsIter->start();
             resultIter->hasNext();
             resultIter->next(), lhsIter->next(), rhsIter->next()) {
            resultIter->curr() = static_cast<real>(lhsIter->curr() == rhsIter->curr());
        }
    }

    void NeqOp::forward() {
        std::unique_ptr<Iter> resultIter = initIter(tensor);
        std::unique_ptr<Iter> lhsIter = initIter(lhs->tensor);
        std::unique_ptr<Iter> rhsIter = initIter(rhs->tensor);

        for (resultIter->start(), lhsIter->start(), rhsIter->start();
             resultIter->hasNext();
             resultIter->next(), lhsIter->next(), rhsIter->next()) {
            resultIter->curr() = static_cast<real>(lhsIter->curr() != rhsIter->curr());
        }
    }

    void LessOp::forward() {
        std::unique_ptr<Iter> resultIter = initIter(tensor);
        std::unique_ptr<Iter> lhsIter = initIter(lhs->tensor);
        std::unique_ptr<Iter> rhsIter = initIter(rhs->tensor);

        for (resultIter->start(), lhsIter->start(), rhsIter->start();
             resultIter->hasNext();
             resultIter->next(), lhsIter->next(), rhsIter->next()) {
            resultIter->curr() = static_cast<real>(lhsIter->curr() < rhsIter->curr());
        }
    }

    void GreaterOp::forward() {
        std::unique_ptr<Iter> resultIter = initIter(tensor);
        std::unique_ptr<Iter> lhsIter = initIter(lhs->tensor);
        std::unique_ptr<Iter> rhsIter = initIter(rhs->tensor);

        for (resultIter->start(), lhsIter->start(), rhsIter->start();
             resultIter->hasNext();
             resultIter->next(), lhsIter->next(), rhsIter->next()) {
            resultIter->curr() = static_cast<real>(lhsIter->curr() > rhsIter->curr());
        }
    }

    void LeqOp::forward() {
        std::unique_ptr<Iter> resultIter = initIter(tensor);
        std::unique_ptr<Iter> lhsIter = initIter(lhs->tensor);
        std::unique_ptr<Iter> rhsIter = initIter(rhs->tensor);

        for (resultIter->start(), lhsIter->start(), rhsIter->start();
             resultIter->hasNext();
             resultIter->next(), lhsIter->next(), rhsIter->next()) {
            resultIter->curr() = static_cast<real>(lhsIter->curr() <= rhsIter->curr());
        }
    }

    void GeqOp::forward() {
        std::unique_ptr<Iter> resultIter = initIter(tensor);
        std::unique_ptr<Iter> lhsIter = initIter(lhs->tensor);
        std::unique_ptr<Iter> rhsIter = initIter(rhs->tensor);

        for (resultIter->start(), lhsIter->start(), rhsIter->start();
             resultIter->hasNext();
             resultIter->next(), lhsIter->next(), rhsIter->next()) {
            resultIter->curr() = static_cast<real>(lhsIter->curr() >= rhsIter->curr());
        }
    }

    void ReluOp::forward() {
        std::unique_ptr<Iter> resultIter = initIter(tensor);
        std::unique_ptr<Iter> operandIter = initIter(operand->tensor);

        for (resultIter->start(), operandIter->start();
             resultIter->hasNext();
             resultIter->next(), operandIter->next()) {
            resultIter->curr() = static_cast<real>(operandIter->curr() > 0.);
        }
    }

    void ReluOp::backward() {
        if (operand->opType != OpType::LEAF) {
            operand->initGrad();
            *operand->grad += *grad * (*operand->tensor > 0.);
        }
    }
}
