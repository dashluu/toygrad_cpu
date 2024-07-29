//
// Created by Trung Luu on 7/12/24.
//

#include "ops.h"
#include "tensor_iter.h"

namespace Toygrad::Tensor {
    void ConstOp::forward() {
        DenseIter iter(tensor);

        for (iter.start(); iter.hasNext(); iter.next()) {
            iter.curr() = c;
        }
    }

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
            real sum = 0.;

            while (rhsIter->hasNext()) {
                if (rhsIter->count() > rhs->shape[rhs->shape.getNumDims() - 1] &&
                    (rhsIter->count() - 1) % rhs->shape[rhs->shape.getNumDims() - 1] == 0) {
                    outIter->curr() = sum;
                    outIter->next();
                    sum = rhsIter->curr();
                } else {
                    sum += rhsIter->curr();
                    // std::cout << rhsIter->curr() << " " << rhsIter->count() << std::endl;
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
        // dx += 1.

        for (lhsGradIter->start(); lhsGradIter->hasNext(); lhsGradIter->next()) {
            lhsGradIter->curr() += 1.;
        }
    }

    void AddOp::forward() {
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

    void ExpOp::forward() {
        IterPtr outIter = initIter(tensor);
        IterPtr opIter = initIter(operand.get());

        for (outIter->start(), opIter->start(); outIter->hasNext(); outIter->next(), opIter->next()) {
            outIter->curr() = std::exp(opIter->curr());
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
            opGradIter->curr() += outGradIter->curr() * std::exp(opIter->curr());
        }
    }

    void RecipOp::forward() {
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
        // dx += dz * 2x

        for (outGradIter->start(), opIter->next(), opGradIter->start();
             outGradIter->hasNext();
             outGradIter->next(), opIter->next(), opGradIter->next()) {
            opGradIter->curr() += outGradIter->curr() * 2 * opIter->curr();
        }
    }

    void EqOp::forward() {
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
        IterPtr outIter = initIter(tensor);
        IterPtr opIter = initIter(operand.get());

        for (outIter->start(), opIter->start(); outIter->hasNext(); outIter->next(), opIter->next()) {
            outIter->curr() = static_cast<real>(opIter->curr() > 0.);
        }
    }

    void ReluOp::backward() {
        operand->initGrad();
        IterPtr outGradIter = initIter(tensor->grad.get());
        IterPtr opIter = initIter(operand.get());
        IterPtr opGradIter = initIter(operand->grad.get());

        // z = max(x, 0.)
        // dx += dz * 1 if x > 0. else 0

        for (outGradIter->start(), opIter->next(), opGradIter->start();
             outGradIter->hasNext();
             outGradIter->next(), opIter->next(), opGradIter->next()) {
            opGradIter->curr() += outGradIter->curr() * static_cast<real>(opIter->curr() > 0.);
        }
    }

    void CopyOp::forward() {
        IterPtr outIter = initIter(tensor);
        IterPtr opIter = initIter(operand.get());

        for (outIter->start(), opIter->start(); outIter->hasNext(); outIter->next(), opIter->next()) {
            outIter->curr() = opIter->curr();
        }
    }
}
