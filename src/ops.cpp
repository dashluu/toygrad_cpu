//
// Created by Trung Luu on 7/12/24.
//

#include "ops.h"
#include "tensor_iter.h"

namespace Toygrad::Tensor {
    void ConstOp::forward() {
        tensor->initVec();
        DenseIter iter(tensor);

        for (iter.start(); iter.hasNext(); iter.next()) {
            iter.curr() = c;
        }
    }

    void ArangeOp::forward() {
        tensor->initVec();
        DenseIter iter(tensor);
        size_t i = 0;

        for (iter.start(); iter.hasNext(); iter.next()) {
            iter.curr() = start + step * i;
            i++;
        }
    }

    void RandintOp::forward() {
        tensor->initVec();
        DenseIter iter(tensor);

        for (iter.start(); iter.hasNext(); iter.next()) {
            iter.curr() = RandGen::randint(min, max);
        }
    }

    void RandnOp::forward() {
        tensor->initVec();
        DenseIter iter(tensor);

        for (iter.start(); iter.hasNext(); iter.next()) {
            iter.curr() = RandGen::randn();
        }
    }

    void FromArrOp::forward() {
        tensor->initVec();
        DenseIter iter(tensor);
        size_t i = 0;

        for (iter.start(); iter.hasNext(); iter.next()) {
            iter.curr() = data[i];
            i++;
        }
    }

    void SumOp::forward() {
        tensor->initVec();
        IterPtr outIter = initIter(tensor);

        if (dim == -1) {
            IterPtr lhsIter = initIter(lhs.get());
            real sum = 0;

            for (lhsIter->start(); lhsIter->hasNext(); lhsIter->next()) {
                sum += lhsIter->curr();
            }

            outIter->start();
            outIter->curr() = sum;
        } else {
            IterPtr rhsIter = initIter(rhs.get());
            rhsIter->start();
            outIter->start();
            real sum = 0.f;

            while (rhsIter->hasNext()) {
                if (rhsIter->count() > rhs->shape[rhs->shape.getNumDims() - 1] &&
                    (rhsIter->count() - 1) % rhs->shape[rhs->shape.getNumDims() - 1] == 0) {
                    outIter->curr() = sum;
                    outIter->next();
                    sum = rhsIter->curr();
                } else {
                    sum += rhsIter->curr();
                }

                rhsIter->next();
            }

            outIter->curr() = sum;
        }
    }

    void SumOp::backward() {
        lhs->initGrad();
        IterPtr lhsGradIter = initIter(lhs->grad.get());

        // z = x1+x2+...+xn
        // dx += 1

        for (lhsGradIter->start(); lhsGradIter->hasNext(); lhsGradIter->next()) {
            lhsGradIter->curr() += 1.f;
        }
    }

    void AddOp::forward() {
        tensor->initVec();
        IterPtr outIter = initIter(tensor);
        IterPtr lhsIter = initIter(lhs.get());
        IterPtr rhsIter = initIter(rhs.get());

        for (outIter->start(), lhsIter->start(), rhsIter->start();
             outIter->hasNext();
             outIter->next(), lhsIter->next(), rhsIter->next()) {
            outIter->curr() = lhsIter->curr() + rhsIter->curr();
        }
    }

    void AddOp::backward() {
        lhs->initGrad();
        rhs->initGrad();
        IterPtr outGradIter = initIter(tensor->grad.get());
        IterPtr lhsGradIter = initIter(lhs->grad.get());
        IterPtr rhsGradIter = initIter(rhs->grad.get());

        for (outGradIter->start(), lhsGradIter->start(), rhsGradIter->start();
             outGradIter->hasNext();
             outGradIter->next(), lhsGradIter->next(), rhsGradIter->next()) {
            lhsGradIter->curr() += outGradIter->curr();
            rhsGradIter->curr() += outGradIter->curr();
        }
    }

    void AddAssignOp::forward() {
        tensor->vec = lhs->vec;
        IterPtr outIter = initIter(tensor);
        IterPtr lhsIter = initIter(lhs.get());
        IterPtr rhsIter = initIter(rhs.get());

        for (outIter->start(), lhsIter->start(), rhsIter->start();
             outIter->hasNext();
             outIter->next(), lhsIter->next(), rhsIter->next()) {
            outIter->curr() = lhsIter->curr() + rhsIter->curr();
        }
    }

    void AddAssignOp::backward() {
        lhs->grad = tensor->grad;
    }

    void SubOp::forward() {
        tensor->initVec();
        IterPtr outIter = initIter(tensor);
        IterPtr lhsIter = initIter(lhs.get());
        IterPtr rhsIter = initIter(rhs.get());

        for (outIter->start(), lhsIter->start(), rhsIter->start();
             outIter->hasNext();
             outIter->next(), lhsIter->next(), rhsIter->next()) {
            outIter->curr() = lhsIter->curr() - rhsIter->curr();
        }
    }

    void SubOp::backward() {
        lhs->initGrad();
        rhs->initGrad();
        IterPtr outGradIter = initIter(tensor->grad.get());
        IterPtr lhsGradIter = initIter(lhs->grad.get());
        IterPtr rhsGradIter = initIter(rhs->grad.get());

        for (outGradIter->start(), lhsGradIter->start(), rhsGradIter->start();
             outGradIter->hasNext();
             outGradIter->next(), lhsGradIter->next(), rhsGradIter->next()) {
            lhsGradIter->curr() += outGradIter->curr();
            rhsGradIter->curr() -= outGradIter->curr();
        }
    }

    void SubAssignOp::forward() {
        tensor->vec = lhs->vec;
        IterPtr outIter = initIter(tensor);
        IterPtr lhsIter = initIter(lhs.get());
        IterPtr rhsIter = initIter(rhs.get());

        for (outIter->start(), lhsIter->start(), rhsIter->start();
             outIter->hasNext();
             outIter->next(), lhsIter->next(), rhsIter->next()) {
            outIter->curr() = lhsIter->curr() - rhsIter->curr();
        }
    }

    void SubAssignOp::backward() {
        lhs->grad = tensor->grad;
    }

    void MulOp::forward() {
        tensor->initVec();
        IterPtr outIter = initIter(tensor);
        IterPtr lhsIter = initIter(lhs.get());
        IterPtr rhsIter = initIter(rhs.get());

        for (outIter->start(), lhsIter->start(), rhsIter->start();
             outIter->hasNext();
             outIter->next(), lhsIter->next(), rhsIter->next()) {
            outIter->curr() = lhsIter->curr() * rhsIter->curr();
        }
    }

    void MulOp::backward() {
        lhs->initGrad();
        rhs->initGrad();
        IterPtr outGradIter = initIter(tensor->grad.get());
        IterPtr lhsIter = initIter(lhs.get());
        IterPtr lhsGradIter = initIter(lhs->grad.get());
        IterPtr rhsIter = initIter(rhs.get());
        IterPtr rhsGradIter = initIter(rhs->grad.get());

        // z = x*y
        // dx += dz*y
        // dy += dx*x

        for (outGradIter->start(), lhsIter->next(), lhsGradIter->start(), rhsIter->start(), rhsGradIter->start();
             outGradIter->hasNext();
             outGradIter->next(), lhsIter->next(), lhsGradIter->next(), rhsIter->next(), rhsGradIter->next()) {
            lhsGradIter->curr() += outGradIter->curr() * rhsIter->curr();
            rhsGradIter->curr() += outGradIter->curr() * lhsIter->curr();
        }
    }

    void MulAssignOp::forward() {
        tensor->vec = lhs->vec;
        IterPtr outIter = initIter(tensor);
        IterPtr lhsIter = initIter(lhs.get());
        IterPtr rhsIter = initIter(rhs.get());

        for (outIter->start(), lhsIter->start(), rhsIter->start();
             outIter->hasNext();
             outIter->next(), lhsIter->next(), rhsIter->next()) {
            outIter->curr() = lhsIter->curr() * rhsIter->curr();
        }
    }

    void MulAssignOp::backward() {
        lhs->grad = tensor->grad;
    }

    void DivOp::forward() {
        tensor->initVec();
        IterPtr outIter = initIter(tensor);
        IterPtr lhsIter = initIter(lhs.get());
        IterPtr rhsIter = initIter(rhs.get());

        for (outIter->start(), lhsIter->start(), rhsIter->start();
             outIter->hasNext();
             outIter->next(), lhsIter->next(), rhsIter->next()) {
            outIter->curr() = lhsIter->curr() / rhsIter->curr();
        }
    }

    void DivOp::backward() {
        lhs->initGrad();
        rhs->initGrad();
        IterPtr outGradIter = initIter(tensor->grad.get());
        IterPtr lhsIter = initIter(lhs.get());
        IterPtr lhsGradIter = initIter(lhs->grad.get());
        IterPtr rhsIter = initIter(rhs.get());
        IterPtr rhsGradIter = initIter(rhs->grad.get());

        // z = x/y
        // dx += dz * (1/y)
        // dy += dz * (-x / y^2)

        for (outGradIter->start(), lhsIter->next(), lhsGradIter->start(), rhsIter->start(), rhsGradIter->start();
             outGradIter->hasNext();
             outGradIter->next(), lhsIter->next(), lhsGradIter->next(), rhsIter->next(), rhsGradIter->next()) {
            lhsGradIter->curr() += outGradIter->curr() / rhsIter->curr();
            rhsGradIter->curr() += outGradIter->curr() * -lhsIter->curr() / (rhsIter->curr() * rhsIter->curr());
        }
    }

    void DivAssignOp::forward() {
        tensor->vec = lhs->vec;
        IterPtr outIter = initIter(tensor);
        IterPtr lhsIter = initIter(lhs.get());
        IterPtr rhsIter = initIter(rhs.get());

        for (outIter->start(), lhsIter->start(), rhsIter->start();
             outIter->hasNext();
             outIter->next(), lhsIter->next(), rhsIter->next()) {
            outIter->curr() = lhsIter->curr() / rhsIter->curr();
        }
    }

    void DivAssignOp::backward() {
        lhs->grad = tensor->grad;
    }

    void PowOp::forward() {
        tensor->initVec();
        IterPtr outIter = initIter(tensor);
        IterPtr opIter = initIter(operand.get());

        for (outIter->start(), opIter->start(); outIter->hasNext(); outIter->next(), opIter->next()) {
            outIter->curr() = pow(opIter->curr(), c);
        }
    }

    void PowOp::backward() {
        operand->initGrad();
        IterPtr outGradIter = initIter(tensor->grad.get());
        IterPtr opIter = initIter(operand.get());
        IterPtr opGradIter = initIter(operand->grad.get());

        // z = x^c
        // dx += dz * c * x^(c-1)

        for (outGradIter->start(), opIter->next(), opGradIter->start();
             outGradIter->hasNext();
             outGradIter->next(), opIter->next(), opGradIter->next()) {
            opGradIter->curr() += outGradIter->curr() * c * pow(opIter->curr(), c - 1);
        }
    }

    void LogOp::forward() {
        tensor->initVec();
        IterPtr outIter = initIter(tensor);
        IterPtr opIter = initIter(operand.get());

        for (outIter->start(), opIter->start(); outIter->hasNext(); outIter->next(), opIter->next()) {
            outIter->curr() = log(opIter->curr());
        }
    }

    void LogOp::backward() {
        operand->initGrad();
        IterPtr outGradIter = initIter(tensor->grad.get());
        IterPtr opIter = initIter(operand.get());
        IterPtr opGradIter = initIter(operand->grad.get());

        // z = log(x)
        // dx += dz * 1 / x

        for (outGradIter->start(), opIter->next(), opGradIter->start();
             outGradIter->hasNext();
             outGradIter->next(), opIter->next(), opGradIter->next()) {
            opGradIter->curr() += outGradIter->curr() / opIter->curr();
        }
    }

    void SinOp::forward() {
        tensor->initVec();
        IterPtr outIter = initIter(tensor);
        IterPtr opIter = initIter(operand.get());

        for (outIter->start(), opIter->start(); outIter->hasNext(); outIter->next(), opIter->next()) {
            outIter->curr() = sin(opIter->curr());
        }
    }

    void SinOp::backward() {
        operand->initGrad();
        IterPtr outGradIter = initIter(tensor->grad.get());
        IterPtr opIter = initIter(operand.get());
        IterPtr opGradIter = initIter(operand->grad.get());

        // z = sin(x)
        // dx += dz * cos(x)

        for (outGradIter->start(), opIter->next(), opGradIter->start();
             outGradIter->hasNext();
             outGradIter->next(), opIter->next(), opGradIter->next()) {
            opGradIter->curr() += outGradIter->curr() * cos(opIter->curr());
        }
    }

    void CosOp::forward() {
        tensor->initVec();
        IterPtr outIter = initIter(tensor);
        IterPtr opIter = initIter(operand.get());

        for (outIter->start(), opIter->start(); outIter->hasNext(); outIter->next(), opIter->next()) {
            outIter->curr() = cos(opIter->curr());
        }
    }

    void CosOp::backward() {
        operand->initGrad();
        IterPtr outGradIter = initIter(tensor->grad.get());
        IterPtr opIter = initIter(operand.get());
        IterPtr opGradIter = initIter(operand->grad.get());

        // z = cos(x)
        // dx += dz * -sin(x)

        for (outGradIter->start(), opIter->next(), opGradIter->start();
             outGradIter->hasNext();
             outGradIter->next(), opIter->next(), opGradIter->next()) {
            opGradIter->curr() += outGradIter->curr() * -sin(opIter->curr());
        }
    }

    void ExpOp::forward() {
        tensor->initVec();
        IterPtr outIter = initIter(tensor);
        IterPtr opIter = initIter(operand.get());

        for (outIter->start(), opIter->start(); outIter->hasNext(); outIter->next(), opIter->next()) {
            outIter->curr() = exp(opIter->curr());
        }
    }

    void ExpOp::backward() {
        operand->initGrad();
        IterPtr outGradIter = initIter(tensor->grad.get());
        IterPtr opIter = initIter(operand.get());
        IterPtr opGradIter = initIter(operand->grad.get());

        // z = e^x
        // dx += dz * e^x

        for (outGradIter->start(), opIter->next(), opGradIter->start();
             outGradIter->hasNext();
             outGradIter->next(), opIter->next(), opGradIter->next()) {
            opGradIter->curr() += outGradIter->curr() * exp(opIter->curr());
        }
    }

    void RecipOp::forward() {
        tensor->initVec();
        IterPtr outIter = initIter(tensor);
        IterPtr opIter = initIter(operand.get());

        for (outIter->start(), opIter->start(); outIter->hasNext(); outIter->next(), opIter->next()) {
            outIter->curr() = c / opIter->curr();
        }
    }

    void RecipOp::backward() {
        operand->initGrad();

        IterPtr outGradIter = initIter(tensor->grad.get());
        IterPtr opIter = initIter(operand.get());
        IterPtr opGradIter = initIter(operand->grad.get());

        // z = c / x
        // dx += dz * (-c / x^2)

        for (outGradIter->start(), opIter->next(), opGradIter->start();
             outGradIter->hasNext();
             outGradIter->next(), opIter->next(), opGradIter->next()) {
            opGradIter->curr() += outGradIter->curr() * -c / (opIter->curr() * opIter->curr());
        }
    }

    void NegOp::forward() {
        tensor->initVec();
        IterPtr outIter = initIter(tensor);
        IterPtr opIter = initIter(operand.get());

        for (outIter->start(), opIter->start(); outIter->hasNext(); outIter->next(), opIter->next()) {
            outIter->curr() = -opIter->curr();
        }
    }

    void NegOp::backward() {
        operand->initGrad();
        IterPtr outGradIter = initIter(tensor->grad.get());
        IterPtr opGradIter = initIter(operand->grad.get());

        // z = -x
        // dx += -dz

        for (outGradIter->start(), opGradIter->start();
             outGradIter->hasNext();
             outGradIter->next(), opGradIter->next()) {
            opGradIter->curr() -= outGradIter->curr();
        }
    }

    void SqOp::forward() {
        tensor->initVec();
        IterPtr outIter = initIter(tensor);
        IterPtr opIter = initIter(operand.get());

        for (outIter->start(), opIter->start(); outIter->hasNext(); outIter->next(), opIter->next()) {
            outIter->curr() = opIter->curr() * opIter->curr();
        }
    }

    void SqOp::backward() {
        operand->initGrad();
        IterPtr outGradIter = initIter(tensor->grad.get());
        IterPtr opIter = initIter(operand.get());
        IterPtr opGradIter = initIter(operand->grad.get());

        // z = x^2
        // dx += dz * 2 * x

        for (outGradIter->start(), opIter->next(), opGradIter->start();
             outGradIter->hasNext();
             outGradIter->next(), opIter->next(), opGradIter->next()) {
            opGradIter->curr() += outGradIter->curr() * 2 * opIter->curr();
        }
    }

    void SqrtOp::forward() {
        tensor->initVec();
        IterPtr outIter = initIter(tensor);
        IterPtr opIter = initIter(operand.get());

        for (outIter->start(), opIter->start(); outIter->hasNext(); outIter->next(), opIter->next()) {
            outIter->curr() = sqrt(opIter->curr());
        }
    }

    void SqrtOp::backward() {
        operand->initGrad();
        IterPtr outGradIter = initIter(tensor->grad.get());
        IterPtr opIter = initIter(operand.get());
        IterPtr opGradIter = initIter(operand->grad.get());

        // z = sqrt(x)
        // dx += dz * 1 / (2 * sqrt(x))

        for (outGradIter->start(), opIter->next(), opGradIter->start();
             outGradIter->hasNext();
             outGradIter->next(), opIter->next(), opGradIter->next()) {
            opGradIter->curr() += outGradIter->curr() / (2 * sqrt(opIter->curr()));
        }
    }

    void AliasOp::forward() {
        tensor->vec = operand->vec;
    }

    void EqOp::forward() {
        tensor->initVec();
        IterPtr outIter = initIter(tensor);
        IterPtr lhsIter = initIter(lhs.get());
        IterPtr rhsIter = initIter(rhs.get());

        for (outIter->start(), lhsIter->start(), rhsIter->start();
             outIter->hasNext();
             outIter->next(), lhsIter->next(), rhsIter->next()) {
            outIter->curr() = static_cast<real>(lhsIter->curr() == rhsIter->curr());
        }
    }

    void NeqOp::forward() {
        tensor->initVec();
        IterPtr outIter = initIter(tensor);
        IterPtr lhsIter = initIter(lhs.get());
        IterPtr rhsIter = initIter(rhs.get());

        for (outIter->start(), lhsIter->start(), rhsIter->start();
             outIter->hasNext();
             outIter->next(), lhsIter->next(), rhsIter->next()) {
            outIter->curr() = static_cast<real>(lhsIter->curr() != rhsIter->curr());
        }
    }

    void LessOp::forward() {
        tensor->initVec();
        IterPtr outIter = initIter(tensor);
        IterPtr lhsIter = initIter(lhs.get());
        IterPtr rhsIter = initIter(rhs.get());

        for (outIter->start(), lhsIter->start(), rhsIter->start();
             outIter->hasNext();
             outIter->next(), lhsIter->next(), rhsIter->next()) {
            outIter->curr() = static_cast<real>(lhsIter->curr() < rhsIter->curr());
        }
    }

    void GreaterOp::forward() {
        tensor->initVec();
        IterPtr outIter = initIter(tensor);
        IterPtr lhsIter = initIter(lhs.get());
        IterPtr rhsIter = initIter(rhs.get());

        for (outIter->start(), lhsIter->start(), rhsIter->start();
             outIter->hasNext();
             outIter->next(), lhsIter->next(), rhsIter->next()) {
            outIter->curr() = static_cast<real>(lhsIter->curr() > rhsIter->curr());
        }
    }

    void LeqOp::forward() {
        tensor->initVec();
        IterPtr outIter = initIter(tensor);
        IterPtr lhsIter = initIter(lhs.get());
        IterPtr rhsIter = initIter(rhs.get());

        for (outIter->start(), lhsIter->start(), rhsIter->start();
             outIter->hasNext();
             outIter->next(), lhsIter->next(), rhsIter->next()) {
            outIter->curr() = static_cast<real>(lhsIter->curr() <= rhsIter->curr());
        }
    }

    void GeqOp::forward() {
        tensor->initVec();
        IterPtr outIter = initIter(tensor);
        IterPtr lhsIter = initIter(lhs.get());
        IterPtr rhsIter = initIter(rhs.get());

        for (outIter->start(), lhsIter->start(), rhsIter->start();
             outIter->hasNext();
             outIter->next(), lhsIter->next(), rhsIter->next()) {
            outIter->curr() = static_cast<real>(lhsIter->curr() >= rhsIter->curr());
        }
    }

    void ReluOp::forward() {
        tensor->initVec();
        IterPtr outIter = initIter(tensor);
        IterPtr opIter = initIter(operand.get());

        for (outIter->start(), opIter->start(); outIter->hasNext(); outIter->next(), opIter->next()) {
            outIter->curr() = static_cast<real>(opIter->curr() > 0.f);
        }
    }

    void ReluOp::backward() {
        operand->initGrad();
        IterPtr outGradIter = initIter(tensor->grad.get());
        IterPtr opIter = initIter(operand.get());
        IterPtr opGradIter = initIter(operand->grad.get());

        // z = max(x, 0)
        // dx += dz * 1 if x > 0 else 0

        for (outGradIter->start(), opIter->next(), opGradIter->start();
             outGradIter->hasNext();
             outGradIter->next(), opIter->next(), opGradIter->next()) {
            opGradIter->curr() += outGradIter->curr() * static_cast<real>(opIter->curr() > 0.f);
        }
    }

    void SigmoidOp::forward() {
        tensor->initVec();
        IterPtr outIter = initIter(tensor);
        IterPtr opIter = initIter(operand.get());

        for (outIter->start(), opIter->start(); outIter->hasNext(); outIter->next(), opIter->next()) {
            outIter->curr() = 1.f / (1.f + exp(-opIter->curr()));
        }
    }

    void SigmoidOp::backward() {
        operand->initGrad();
        IterPtr outGradIter = initIter(tensor->grad.get());
        IterPtr opIter = initIter(operand.get());
        IterPtr opGradIter = initIter(operand->grad.get());

        // z = 1/(1+exp(-x))
        // dx += dz*z*(1-z)

        for (outGradIter->start(), opIter->next(), opGradIter->start();
             outGradIter->hasNext();
             outGradIter->next(), opIter->next(), opGradIter->next()) {
            real sigm = 1.f / (1.f + exp(-opIter->curr()));
            opGradIter->curr() += outGradIter->curr() * sigm * (1 - sigm);
        }
    }

    void CopyOp::forward() {
        tensor->initVec();
        IterPtr outIter = initIter(tensor);
        IterPtr opIter = initIter(operand.get());

        for (outIter->start(), opIter->start(); outIter->hasNext(); outIter->next(), opIter->next()) {
            outIter->curr() = opIter->curr();
        }
    }
}
