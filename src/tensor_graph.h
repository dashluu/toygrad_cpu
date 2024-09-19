//
// Created by Trung Luu on 7/26/24.
//

#pragma once

#include <unordered_set>
#include "tensor.h"

// Computational graph
namespace Toygrad::Tensor {
    class TensorGraph {
        std::vector<TensorPtr> tensors;
        TensorPtr root;

        TensorGraph() = default;

        void recurSort(const TensorPtr &tensor, std::unordered_set<size_t> visited);

    public:
        explicit TensorGraph(const TensorPtr &root, const bool doSort = true): root(root) {
            if (doSort) {
                sort();
            }
        }

        void sort();

        TensorPtr getRoot() const { return root; }

        void forward() const;

        void backward() const;
    };
}
