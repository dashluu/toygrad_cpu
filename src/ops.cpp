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
        IterPtr opIter = initIter(operand.get());
        real sum = 0.f;

        if (dim == -1) {
            for (opIter->start(); opIter->hasNext(); opIter->next()) {
                sum += opIter->curr();
            }

            outIter->start();
            outIter->curr() = sum;
        } else {
            opIter->start();
            outIter->start();

            while (opIter->hasNext()) {
                if (opIter->count() > operand->shape[operand->shape.getNumDims() - 1] &&
                    (opIter->count() - 1) % operand->shape[operand->shape.getNumDims() - 1] == 0) {
                    outIter->curr() = sum;
                    outIter->next();
                    sum = opIter->curr();
                } else {
                    sum += opIter->curr();
                }

                opIter->next();
            }

            outIter->curr() = sum;
        }
    }

    void SumOp::backward() {
        tensor->initGrad(1.f);
        operand->initGrad();
        IterPtr outGradIter = initIter(tensor->grad.get());
        IterPtr opGradIter = initIter(operand->grad.get());

        // z = x1+x2+...+xn
        // dx += dz * 1.

        if (dim == -1) {
            for (opGradIter->start(), outGradIter->start(); opGradIter->hasNext(); opGradIter->next()) {
                opGradIter->curr() += outGradIter->curr();
            }
        } else {
            opGradIter->start();
            outGradIter->start();
            size_t lastDim = operand->shape[operand->shape.getNumDims() - 1];

            while (opGradIter->hasNext()) {
                opGradIter->curr() += outGradIter->curr();

                if (opGradIter->count() > lastDim && (opGradIter->count() - 1) % lastDim == 0) {
                    outGradIter->next();
                }

                opGradIter->next();
            }
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
        IterPtr outIter = initIter(tensor);
        IterPtr opIter = initIter(operand.get());

        for (outIter->start(), opIter->start(); outIter->hasNext(); outIter->next(), opIter->next()) {
            outIter->curr() += opIter->curr();
        }
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
        IterPtr outIter = initIter(tensor);
        IterPtr opIter = initIter(operand.get());

        for (outIter->start(), opIter->start(); outIter->hasNext(); outIter->next(), opIter->next()) {
            outIter->curr() -= opIter->curr();
        }
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

        for (outGradIter->start(), lhsIter->start(), lhsGradIter->start(), rhsIter->start(), rhsGradIter->start();
             outGradIter->hasNext();
             outGradIter->next(), lhsIter->next(), lhsGradIter->next(), rhsIter->next(), rhsGradIter->next()) {
            lhsGradIter->curr() += outGradIter->curr() * rhsIter->curr();
            rhsGradIter->curr() += outGradIter->curr() * lhsIter->curr();
        }
    }

    void MulAssignOp::forward() {
        IterPtr outIter = initIter(tensor);
        IterPtr opIter = initIter(operand.get());

        for (outIter->start(), opIter->start(); outIter->hasNext(); outIter->next(), opIter->next()) {
            outIter->curr() *= opIter->curr();
        }
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

        for (outGradIter->start(), lhsIter->start(), lhsGradIter->start(), rhsIter->start(), rhsGradIter->start();
             outGradIter->hasNext();
             outGradIter->next(), lhsIter->next(), lhsGradIter->next(), rhsIter->next(), rhsGradIter->next()) {
            lhsGradIter->curr() += outGradIter->curr() / rhsIter->curr();
            rhsGradIter->curr() += outGradIter->curr() * -lhsIter->curr() / (rhsIter->curr() * rhsIter->curr());
        }
    }

    void DivAssignOp::forward() {
        IterPtr outIter = initIter(tensor);
        IterPtr opIter = initIter(operand.get());

        for (outIter->start(), opIter->start(); outIter->hasNext(); outIter->next(), opIter->next()) {
            outIter->curr() /= opIter->curr();
        }
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

        for (outGradIter->start(), opIter->start(), opGradIter->start();
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

        for (outGradIter->start(), opIter->start(), opGradIter->start();
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

        for (outGradIter->start(), opIter->start(), opGradIter->start();
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

        for (outGradIter->start(), opIter->start(), opGradIter->start();
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

        for (outGradIter->start(), opIter->start(), opGradIter->start();
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

        for (outGradIter->start(), opIter->start(), opGradIter->start();
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

        for (outGradIter->start(), opIter->start(), opGradIter->start();
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

        for (outGradIter->start(), opIter->start(), opGradIter->start();
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

    void MaxOp::forward() {
        tensor->initVec();
        IterPtr outIter = initIter(tensor);
        IterPtr opIter = initIter(operand.get());
        opIter->start();

        if (!opIter->hasNext())
            return;

        real max = opIter->curr();
        opIter->next();

        if (dim == -1) {
            while (opIter->hasNext()) {
                if (opIter->curr() > max)
                    max = opIter->curr();
                opIter->next();
            }

            outIter->start();
            outIter->curr() = max;
        } else {
            opIter->start();
            outIter->start();

            while (opIter->hasNext()) {
                if (opIter->count() > operand->shape[operand->shape.getNumDims() - 1] &&
                    (opIter->count() - 1) % operand->shape[operand->shape.getNumDims() - 1] == 0) {
                    outIter->curr() = max;
                    outIter->next();
                    max = opIter->curr();
                } else if (opIter->curr() > max) {
                    max = opIter->curr();
                }

                opIter->next();
            }

            outIter->curr() = max;
        }
    }

    void MaxOp::backward() {
        operand->initGrad();
        IterPtr outGradIter = initIter(tensor->grad.get());
        IterPtr opGradIter = initIter(operand->grad.get());

        // z = max(x1,x2,...,xn)
        // dx += dz * [1. if xi == z else 0.]

        if (dim == -1) {
            for (opGradIter->start(), outGradIter->start(); opGradIter->hasNext(); opGradIter->next()) {
                opGradIter->curr() += outGradIter->curr();
            }
        } else {
            IterPtr outIter = initIter(tensor);
            IterPtr opIter = initIter(operand.get());
            outIter->start();
            outGradIter->start();
            opIter->start();
            opGradIter->start();
            size_t lastDim = operand->shape[operand->shape.getNumDims() - 1];

            while (opGradIter->hasNext()) {
                if (outIter->curr() == opIter->curr())
                    opGradIter->curr() += outGradIter->curr();

                if (opGradIter->count() > lastDim && (opGradIter->count() - 1) % lastDim == 0) {
                    outIter->next();
                    outGradIter->next();
                }

                opIter->next();
                opGradIter->next();
            }
        }
    }

    void MinOp::forward() {
        tensor->initVec();
        IterPtr outIter = initIter(tensor);
        IterPtr opIter = initIter(operand.get());
        opIter->start();

        if (!opIter->hasNext())
            return;

        real min = opIter->curr();
        opIter->next();

        if (dim == -1) {
            while (opIter->hasNext()) {
                if (opIter->curr() < min)
                    min = opIter->curr();
                opIter->next();
            }

            outIter->start();
            outIter->curr() = min;
        } else {
            opIter->start();
            outIter->start();

            while (opIter->hasNext()) {
                if (opIter->count() > operand->shape[operand->shape.getNumDims() - 1] &&
                    (opIter->count() - 1) % operand->shape[operand->shape.getNumDims() - 1] == 0) {
                    outIter->curr() = min;
                    outIter->next();
                    min = opIter->curr();
                } else if (opIter->curr() < min) {
                    min = opIter->curr();
                }

                opIter->next();
            }

            outIter->curr() = min;
        }
    }

    void MinOp::backward() {
        operand->initGrad();
        IterPtr outGradIter = initIter(tensor->grad.get());
        IterPtr opGradIter = initIter(operand->grad.get());

        // z = min(x1,x2,...,xn)
        // dx += dz * [1. if xi == z else 0.]

        if (dim == -1) {
            for (opGradIter->start(), outGradIter->start(); opGradIter->hasNext(); opGradIter->next()) {
                opGradIter->curr() += outGradIter->curr();
            }
        } else {
            IterPtr outIter = initIter(tensor);
            IterPtr opIter = initIter(operand.get());
            outIter->start();
            outGradIter->start();
            opIter->start();
            opGradIter->start();
            size_t lastDim = operand->shape[operand->shape.getNumDims() - 1];

            while (opGradIter->hasNext()) {
                if (outIter->curr() == opIter->curr())
                    opGradIter->curr() += outGradIter->curr();

                if (opGradIter->count() > lastDim && (opGradIter->count() - 1) % lastDim == 0) {
                    outIter->next();
                    outGradIter->next();
                }

                opIter->next();
                opGradIter->next();
            }
        }
    }

    void PermOp::forward() {
        tensor->vec = operand->vec;
    }

    void PermOp::backward() {
        operand->initGrad();
        IterPtr outGradIter = initIter(tensor->grad.get());
        IterPtr opGradIter = initIter(operand->grad.get());

        for (outGradIter->start(), opGradIter->start();
             outGradIter->hasNext();
             outGradIter->next(), opGradIter->next()) {
            opGradIter->curr() = outGradIter->curr();
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

        for (outGradIter->start(), opIter->start(), opGradIter->start();
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

        // z = 1 / (1 + exp(-x))
        // dx += dz * z * (1 - z)

        for (outGradIter->start(), opIter->start(), opGradIter->start();
             outGradIter->hasNext();
             outGradIter->next(), opIter->next(), opGradIter->next()) {
            real sigmoid = 1 / (1 + exp(-opIter->curr()));
            opGradIter->curr() += outGradIter->curr() * sigmoid * (1 - sigmoid);
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

    void MatmulOp::forward() {
        tensor->initVec();
        IterPtr outIter = initIter(tensor);
        IterPtr lhsIter = initIter(lhs.get());
        IterPtr rhsIter = initIter(rhs.get());
        size_t numDims = tensor->shape.getNumDims();
        real sum;
        outIter->start();
        lhsIter->start();
        rhsIter->start();
        lhsIter->save();
        rhsIter->save();

        while (outIter->hasNext()) {
            for (size_t i = 0; i < lhs->shape[numDims - 2]; i++) {
                for (size_t j = 0; j < rhs->shape[numDims - 2]; j++) {
                    sum = 0;

                    for (size_t k = 0; k < rhs->shape[numDims - 1]; k++, lhsIter->next(), rhsIter->next()) {
                        sum += lhsIter->curr() * rhsIter->curr();
                    }

                    outIter->curr() = sum;
                    outIter->next();

                    if (j < rhs->shape[numDims - 2] - 1) {
                        lhsIter->restore();
                    }

                    lhsIter->save();
                }

                if (i < lhs->shape[numDims - 2] - 1) {
                    rhsIter->restore();
                }

                rhsIter->save();
            }
        }
    }

    void MatmulOp::backward() {
        lhs->initGrad();
        rhs->initGrad();
        // rhs already switches the last two dimensions so no need to do transpose here
        lhs->grad = tensor->grad->matmul(rhs);
        rhs->grad = lhs->T(lhs->shape.getNumDims() - 2)->matmul(tensor->grad);
    }
}
