//
// Created by Trung Luu on 7/26/24.
//

#include "cgraph.h"
#include "ops.h"

namespace Toygrad::Tensor {
    void CGraph::backprop(const TensorPtr &root) {
        std::vector<TensorPtr> stack;
        stack.push_back(root);

        while (!stack.empty()) {
            TensorPtr tensor = stack.back();
            stack.pop_back();
            tensor->op->backward();

            if (tensor->op->opType == OpType::UN_OP) {
                auto op = dynamic_cast<UnOp *>(tensor->op);
                auto operand = op->operand;
                if (operand) stack.push_back(operand);
            } else if (tensor->op->opType == OpType::BIN_OP) {
                auto op = dynamic_cast<BinOp *>(tensor->op);
                auto lhs = op->lhs;
                auto rhs = op->rhs;
                if (lhs) stack.push_back(lhs);
                if (rhs) stack.push_back(rhs);
            }
        }
    }
}
