//
// Created by Trung Luu on 7/26/24.
//

#include "cgraph.h"

#include <ranges>
#include "ops.h"

namespace Toygrad::Tensor {
    void CGraph::backprop(const TensorPtr &root) {
        std::vector<TensorPtr> stack;
        stack.push_back(root);

        while (!stack.empty()) {
            TensorPtr tensor = stack.back();
            stack.pop_back();

            for (auto &op: std::ranges::reverse_view(tensor->ops)) {
                op->backward();

                if (op->opType == OpType::UN_OP) {
                    auto unOp = dynamic_cast<UnOp *>(op);
                    stack.push_back(unOp->operand);
                } else if (op->opType == OpType::BIN_OP) {
                    auto binOp = dynamic_cast<BinOp *>(op);
                    stack.push_back(binOp->lhs);
                    stack.push_back(binOp->rhs);
                }
            }
        }
    }
}
