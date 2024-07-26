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

    void SumOp::forward() {
        std::unique_ptr<Iter> resultIter = initIter(tensor);
        std::unique_ptr<Iter> operandIter = initIter(operand.get());
        real sum = 0;

        for (operandIter->start(); operandIter->hasNext(); operandIter->next()) {
            sum += operandIter->curr();
        }

        resultIter->start();
        resultIter->curr() = sum;
    }

    void SumOp::backward() {
        operand->initGrad();
        operand->grad->addAssign(tensor->grad->mul(1.));
    }

    void AddOp::forward() {
        std::unique_ptr<Iter> resultIter = initIter(tensor);
        std::unique_ptr<Iter> lhsIter = initIter(lhs.get());
        std::unique_ptr<Iter> rhsIter = initIter(rhs.get());

        for (resultIter->start(), lhsIter->start(), rhsIter->start();
             resultIter->hasNext();
             resultIter->next(), lhsIter->next(), rhsIter->next()) {
            resultIter->curr() = lhsIter->curr() + rhsIter->curr();
        }
    }

    void AddOp::backward() {
        lhs->initGrad();
        lhs->grad->addAssign(tensor->grad->mul(1.));
        rhs->initGrad();
        rhs->grad->addAssign(tensor->grad->mul(1.));
    }

    void AddAssignOp::forward() {
        std::unique_ptr<Iter> resultIter = initIter(tensor);
        std::unique_ptr<Iter> lhsIter = initIter(lhs.get());
        std::unique_ptr<Iter> rhsIter = initIter(rhs.get());

        for (resultIter->start(), lhsIter->start(), rhsIter->start();
             resultIter->hasNext();
             resultIter->next(), lhsIter->next(), rhsIter->next()) {
            resultIter->curr() = lhsIter->curr() + rhsIter->curr();
        }
    }

    void AddAssignOp::backward() {
        lhs->grad = tensor->grad;
    }

    void SubOp::forward() {
        std::unique_ptr<Iter> resultIter = initIter(tensor);
        std::unique_ptr<Iter> lhsIter = initIter(lhs.get());
        std::unique_ptr<Iter> rhsIter = initIter(rhs.get());

        for (resultIter->start(), lhsIter->start(), rhsIter->start();
             resultIter->hasNext();
             resultIter->next(), lhsIter->next(), rhsIter->next()) {
            resultIter->curr() = lhsIter->curr() - rhsIter->curr();
        }
    }

    void SubOp::backward() {
        lhs->initGrad();
        lhs->grad->addAssign(tensor->grad->mul(1.));
        rhs->initGrad();
        rhs->grad->addAssign(tensor->grad->mul(-1.));
    }

    void SubAssignOp::forward() {
        std::unique_ptr<Iter> resultIter = initIter(tensor);
        std::unique_ptr<Iter> lhsIter = initIter(lhs.get());
        std::unique_ptr<Iter> rhsIter = initIter(rhs.get());

        for (resultIter->start(), lhsIter->start(), rhsIter->start();
             resultIter->hasNext();
             resultIter->next(), lhsIter->next(), rhsIter->next()) {
            resultIter->curr() = lhsIter->curr() - rhsIter->curr();
        }
    }

    void SubAssignOp::backward() {
        lhs->grad = tensor->grad;
    }

    void MulOp::forward() {
        std::unique_ptr<Iter> resultIter = initIter(tensor);
        std::unique_ptr<Iter> lhsIter = initIter(lhs.get());
        std::unique_ptr<Iter> rhsIter = initIter(rhs.get());

        for (resultIter->start(), lhsIter->start(), rhsIter->start();
             resultIter->hasNext();
             resultIter->next(), lhsIter->next(), rhsIter->next()) {
            resultIter->curr() = lhsIter->curr() * rhsIter->curr();
        }
    }

    void MulOp::backward() {
        lhs->initGrad();
        lhs->grad->addAssign(tensor->grad->mul(rhs));
        rhs->initGrad();
        rhs->grad->addAssign(tensor->grad->mul(lhs));
    }

    void MulAssignOp::forward() {
        std::unique_ptr<Iter> resultIter = initIter(tensor);
        std::unique_ptr<Iter> lhsIter = initIter(lhs.get());
        std::unique_ptr<Iter> rhsIter = initIter(rhs.get());

        for (resultIter->start(), lhsIter->start(), rhsIter->start();
             resultIter->hasNext();
             resultIter->next(), lhsIter->next(), rhsIter->next()) {
            resultIter->curr() = lhsIter->curr() * rhsIter->curr();
        }
    }

    void MulAssignOp::backward() {
        lhs->grad = tensor->grad;
    }

    void DivOp::forward() {
        std::unique_ptr<Iter> resultIter = initIter(tensor);
        std::unique_ptr<Iter> lhsIter = initIter(lhs.get());
        std::unique_ptr<Iter> rhsIter = initIter(rhs.get());

        for (resultIter->start(), lhsIter->start(), rhsIter->start();
             resultIter->hasNext();
             resultIter->next(), lhsIter->next(), rhsIter->next()) {
            resultIter->curr() = lhsIter->curr() / rhsIter->curr();
        }
    }

    void DivOp::backward() {
        lhs->initGrad();
        // z = x/y
        // dx = dz * (1/y)
        lhs->grad->addAssign(tensor->grad->mul(rhs->recip()));
        rhs->initGrad();
        // z = x/y
        // dy = dz * (-x / y^2)
        rhs->grad->addAssign(tensor->grad->mul(lhs->neg()->div(rhs->sq())));
    }

    void DivAssignOp::forward() {
        std::unique_ptr<Iter> resultIter = initIter(tensor);
        std::unique_ptr<Iter> lhsIter = initIter(lhs.get());
        std::unique_ptr<Iter> rhsIter = initIter(rhs.get());

        for (resultIter->start(), lhsIter->start(), rhsIter->start();
             resultIter->hasNext();
             resultIter->next(), lhsIter->next(), rhsIter->next()) {
            resultIter->curr() = lhsIter->curr() / rhsIter->curr();
        }
    }

    void DivAssignOp::backward() {
        lhs->grad = tensor->grad;
    }

    void ExpOp::forward() {
        std::unique_ptr<Iter> resultIter = initIter(tensor);
        std::unique_ptr<Iter> operandIter = initIter(operand.get());

        for (resultIter->start(), operandIter->start();
             resultIter->hasNext();
             resultIter->next(), operandIter->next()) {
            resultIter->curr() = std::exp(operandIter->curr());
        }
    }

    void ExpOp::backward() {
        operand->initGrad();
        // z = e^x
        // dx = dz * e^x
        operand->grad->addAssign(tensor->grad->mul(operand->exp()));
    }

    void RecipOp::forward() {
        std::unique_ptr<Iter> resultIter = initIter(tensor);
        std::unique_ptr<Iter> operandIter = initIter(operand.get());

        for (resultIter->start(), operandIter->start();
             resultIter->hasNext();
             resultIter->next(), operandIter->next()) {
            resultIter->curr() = c / operandIter->curr();
        }
    }

    void RecipOp::backward() {
        operand->initGrad();
        // z = c / x
        // dx = dz * (-c / x^2)
        operand->addAssign(tensor->grad->mul(operand->sq()->recip(-c)));
    }

    void NegOp::forward() {
        std::unique_ptr<Iter> resultIter = initIter(tensor);
        std::unique_ptr<Iter> operandIter = initIter(operand.get());

        for (resultIter->start(), operandIter->start();
             resultIter->hasNext();
             resultIter->next(), operandIter->next()) {
            resultIter->curr() = -operandIter->curr();
        }
    }

    void NegOp::backward() {
        operand->initGrad();
        operand->addAssign(tensor->grad->mul(-1.));
    }

    void SqOp::forward() {
        std::unique_ptr<Iter> resultIter = initIter(tensor);
        std::unique_ptr<Iter> operandIter = initIter(operand.get());

        for (resultIter->start(), operandIter->start();
             resultIter->hasNext();
             resultIter->next(), operandIter->next()) {
            resultIter->curr() = operandIter->curr() * operandIter->curr();
        }
    }

    void SqOp::backward() {
        operand->initGrad();
        // z = x^2
        // dx = dz * 2x
        operand->addAssign(tensor->grad->mul(operand->mul(2)));
    }

    void EqOp::forward() {
        std::unique_ptr<Iter> resultIter = initIter(tensor);
        std::unique_ptr<Iter> lhsIter = initIter(lhs.get());
        std::unique_ptr<Iter> rhsIter = initIter(rhs.get());

        for (resultIter->start(), lhsIter->start(), rhsIter->start();
             resultIter->hasNext();
             resultIter->next(), lhsIter->next(), rhsIter->next()) {
            resultIter->curr() = static_cast<real>(lhsIter->curr() == rhsIter->curr());
        }
    }

    void NeqOp::forward() {
        std::unique_ptr<Iter> resultIter = initIter(tensor);
        std::unique_ptr<Iter> lhsIter = initIter(lhs.get());
        std::unique_ptr<Iter> rhsIter = initIter(rhs.get());

        for (resultIter->start(), lhsIter->start(), rhsIter->start();
             resultIter->hasNext();
             resultIter->next(), lhsIter->next(), rhsIter->next()) {
            resultIter->curr() = static_cast<real>(lhsIter->curr() != rhsIter->curr());
        }
    }

    void LessOp::forward() {
        std::unique_ptr<Iter> resultIter = initIter(tensor);
        std::unique_ptr<Iter> lhsIter = initIter(lhs.get());
        std::unique_ptr<Iter> rhsIter = initIter(rhs.get());

        for (resultIter->start(), lhsIter->start(), rhsIter->start();
             resultIter->hasNext();
             resultIter->next(), lhsIter->next(), rhsIter->next()) {
            resultIter->curr() = static_cast<real>(lhsIter->curr() < rhsIter->curr());
        }
    }

    void GreaterOp::forward() {
        std::unique_ptr<Iter> resultIter = initIter(tensor);
        std::unique_ptr<Iter> lhsIter = initIter(lhs.get());
        std::unique_ptr<Iter> rhsIter = initIter(rhs.get());

        for (resultIter->start(), lhsIter->start(), rhsIter->start();
             resultIter->hasNext();
             resultIter->next(), lhsIter->next(), rhsIter->next()) {
            resultIter->curr() = static_cast<real>(lhsIter->curr() > rhsIter->curr());
        }
    }

    void LeqOp::forward() {
        std::unique_ptr<Iter> resultIter = initIter(tensor);
        std::unique_ptr<Iter> lhsIter = initIter(lhs.get());
        std::unique_ptr<Iter> rhsIter = initIter(rhs.get());

        for (resultIter->start(), lhsIter->start(), rhsIter->start();
             resultIter->hasNext();
             resultIter->next(), lhsIter->next(), rhsIter->next()) {
            resultIter->curr() = static_cast<real>(lhsIter->curr() <= rhsIter->curr());
        }
    }

    void GeqOp::forward() {
        std::unique_ptr<Iter> resultIter = initIter(tensor);
        std::unique_ptr<Iter> lhsIter = initIter(lhs.get());
        std::unique_ptr<Iter> rhsIter = initIter(rhs.get());

        for (resultIter->start(), lhsIter->start(), rhsIter->start();
             resultIter->hasNext();
             resultIter->next(), lhsIter->next(), rhsIter->next()) {
            resultIter->curr() = static_cast<real>(lhsIter->curr() >= rhsIter->curr());
        }
    }

    void ReluOp::forward() {
        std::unique_ptr<Iter> resultIter = initIter(tensor);
        std::unique_ptr<Iter> operandIter = initIter(operand.get());

        for (resultIter->start(), operandIter->start();
             resultIter->hasNext();
             resultIter->next(), operandIter->next()) {
            resultIter->curr() = static_cast<real>(operandIter->curr() > 0.);
        }
    }

    void ReluOp::backward() {
        operand->initGrad();
        operand->grad->addAssign(tensor->grad->mul(operand->gt(0.)));
    }

    void CopyOp::forward() {
        std::unique_ptr<Iter> resultIter = initIter(tensor);
        std::unique_ptr<Iter> operandIter = initIter(operand.get());

        for (resultIter->start(), operandIter->start();
             resultIter->hasNext();
             resultIter->next(), operandIter->next()) {
            resultIter->curr() = operandIter->curr();
        }
    }
}
