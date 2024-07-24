//
// Created by Trung Luu on 7/20/24.
//

#include <cmath>
#include "tensor_indexer.h"
#include "ops.h"

namespace Toygrad::Tensor {
    Tensor *TensorIndexer::at(const std::vector<size_t> &idx) {
        auto ranges = std::vector<Range>();

        for (size_t i: idx) {
            ranges.push_back({i, i + 1, 1});
        }

        for (size_t i = idx.size(); i < tensor->shape.getNumDims(); i++) {
            ranges.push_back({0, tensor->shape[i], 1});
        }

        return at(ranges);
    }

    Tensor *TensorIndexer::at(const std::vector<Range> &ranges) {
        Shape shape;
        shape.offset = tensor->shape.offset;
        shape.root = &tensor->shape;
        shape.ranges = ranges;

        for (size_t i = 0; i < ranges.size(); i++) {
            shape.offset += ranges[i].beg * tensor->shape.strides[i];
        }

        for (size_t i = 0; i < ranges.size(); i++) {
            size_t dim = ceil(static_cast<real>(ranges[i].end - ranges[i].beg) / ranges[i].step);
            shape.view.push_back(dim);
            shape.strides.push_back(tensor->shape.strides[i] * ranges[i].step);
        }

        return new Tensor(shape, tensor->vec);
    }
}
