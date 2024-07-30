//
// Created by Trung Luu on 7/23/24.
//

#include "tensor_iter.h"

namespace Toygrad::Tensor {
    void SparseIter::next() {
        state.counter++;

        if (state.counter > tensor->getShape().getSize()) {
            return;
        }

        auto &shape = tensor->getShape();
        bool flag;

        do {
            flag = state.rotator[state.ridx] + 1 < shape.view[state.ridx];

            if (!flag) {
                state.ridx--;
            }
        } while (!flag);

        state.rotator[state.ridx]++;
        state.elmIdx = offset;

        for (size_t i = state.ridx + 1; i < state.rotator.size(); i++) {
            state.rotator[i] = 0;
        }

        for (size_t i = 0; i < state.rotator.size(); i++) {
            state.elmIdx += state.rotator[i] * tensor->getShape().strides[i];
        }

        state.ridx = state.rotator.size() - 1;
    }

    IterPtr initIter(Tensor *tensor) {
        if (tensor->isContiguous()) {
            return std::make_unique<DenseIter>(tensor);
        }

        return std::make_unique<SparseIter>(tensor);
    }
}
