//
// Created by Trung Luu on 7/26/24.
//

#pragma once

#include <unordered_set>
#include "tensor.h"

// Computational graph
namespace Toygrad::Tensor {
    class TensorGraph {
        std::vector<Tensor *> tensors;
        Tensor *root = nullptr;

        TensorGraph() = default;

        void recurSort(Tensor *tensor, std::unordered_set<size_t> visited);

        void sort();

    public:
        explicit TensorGraph(Tensor *root): root(root) {
            sort();
        }

        void forward() const;

        void backward() const;
    };
}
