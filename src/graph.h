//
// Created by Trung Luu on 7/20/24.
//

#pragma once

#include <ranges>
#include "ops.h"

namespace Toygrad::Tensor {
    class Graph {
        std::vector<Op *> ops = std::vector<Op *>();

        static Graph &inst() {
            static Graph graph;
            return graph;
        }

    public:
        static void addOp(Op *op) {
            inst().ops.push_back(op);
        }

        static void cleanUp() {
            for (auto iter: inst().ops) {
                delete iter;
            }
        }
    };
}
