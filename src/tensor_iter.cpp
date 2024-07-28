//
// Created by Trung Luu on 7/23/24.
//

#include "tensor_iter.h"

namespace Toygrad::Tensor {
    void SparseIter::next() {
        counter++;

        if (counter > tensor->getShape().getSize()) {
            return;
        }

        auto &shape = tensor->getShape();
        bool flag;

        do {
            flag = rotator[ridx] + 1 < shape.view[ridx];

            if (!flag) {
                ridx--;
            }
        } while (!flag);

        rotator[ridx]++;
        elmIdx = offset;

        for (size_t i = ridx + 1; i < rotator.size(); i++) {
            rotator[i] = 0;
        }

        for (size_t i = 0; i < rotator.size(); i++) {
            elmIdx += rotator[i] * tensor->getShape().strides[i];
        }

        ridx = rotator.size() - 1;
    }

    IterPtr initIter(Tensor *tensor) {
        if (tensor->isContiguous()) {
            return std::make_unique<DenseIter>(tensor);
        }

        return std::make_unique<SparseIter>(tensor);
    }
}
