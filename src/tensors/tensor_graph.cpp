//
// Created by Trung Luu on 7/26/24.
//

#include <ranges>
#include "tensor_graph.h"
#include "ops.h"

namespace Toygrad::Tensor {
    void TensorGraph::recurSort(Tensor *tensor, std::unordered_set<size_t> visited) {
        if (!visited.contains(tensor->id)) {
            visited.insert(tensor->id);

            for (auto &op: tensor->ops) {
                if (op->opType == OpType::UN_OP) {
                    auto unOp = dynamic_cast<UnOp *>(op);
                    recurSort(unOp->operand.get(), visited);
                } else if (op->opType == OpType::BIN_OP) {
                    auto binOp = dynamic_cast<BinOp *>(op);
                    recurSort(binOp->lhs.get(), visited);
                    recurSort(binOp->rhs.get(), visited);
                }
            }

            tensors.push_back(tensor);
        }
    }

    void TensorGraph::sort() {
        std::unordered_set<size_t> visited;
        recurSort(root, visited);
    }

    void TensorGraph::forward() const {
        for (auto &tensor: tensors) {
            for (auto &op: tensor->ops) {
                op->forward();
            }
        }
    }

    void TensorGraph::backward() const {
        for (auto &tensor: std::ranges::reverse_view(tensors)) {
            for (auto &op: std::ranges::reverse_view(tensor->ops)) {
                op->backward();
            }
        }
    }
}
