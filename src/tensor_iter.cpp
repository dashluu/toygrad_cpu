//
// Created by Trung Luu on 7/23/24.
//

#include "tensor_iter.h"

namespace Toygrad::Tensor {
    void SparseIter::next() {
        auto &shape = tensor->getShape();
        auto &dst = shape.dst;
        auto &sst = shape.sst;
        bool flag = false;

        while (nIdx > 0 && !flag) {
            auto tmp = shape.root->offset;

            for (size_t i = 0; i < nIdx; i++) {
                tmp += shape.rng[i].beg * shape.root->dst[i];
            }

            flag = nIndices[nIdx] + 1 < sst[nIdx - 1] &&
                   offset + (nIndices[nIdx] + 1) * dst[nIdx] < tmp + shape.root->dst[nIdx - 1];

            if (!flag) {
                nIdx--;
            }
        }

        nIndices[nIdx]++;
        elmIdx = offset;

        for (size_t i = nIdx + 1; i < nIndices.size(); i++) {
            nIndices[i] = 0;
        }

        for (size_t i = 0; i < nIndices.size(); i++) {
            elmIdx += nIndices[i] * tensor->getShape().dst[i];
        }

        nIdx = nIndices.size() - 1;
        counter++;
    }

    std::unique_ptr<Iter> initIter(Tensor *tensor) {
        if (tensor->isContiguous()) {
            return std::make_unique<DenseIter>(tensor);
        }

        return std::make_unique<SparseIter>(tensor);
    }
}
