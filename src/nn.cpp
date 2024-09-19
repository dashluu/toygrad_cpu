//
// Created by Trung Luu on 9/18/24.
//

#include "nn.h"
#include "ops.h"

namespace Toygrad::NN {
    Tensor::TensorPtr Module::init(const std::vector<Tensor::TensorPtr> &x) {
        if (graph == nullptr) {
            for (size_t i = 0; i < x.size(); i++) {
                input[i] = x[i]->diffAlias();
            }

            graph = std::make_unique<Tensor::TensorGraph>(forwardFunc(input), false);
        } else {
            for (size_t i = 0; i < x.size(); i++) {
                // Clear ops and references to other nodes from previous input nodes
                input[i]->clearOps();
                input[i]->shape = x[i]->shape;
                // No need to copy vec addresses since forward propagation does it any way
                input[i]->ops.push_back(new Tensor::DiffAliasOp(x[i], input[i].get()));
            }
        }

        return graph->getRoot();
    }
}
