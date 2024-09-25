//
// Created by Trung Luu on 9/19/24.
//

#include <sstream>
#include <graphviz/gvc.h>
#include "tensor_graph.h"
#include "tensor_draw.h"

#include "ops.h"
#include "assert/str_assert.h"

namespace Toygrad::Tensor {
    char *TensorDraw::strToCharPtr(const std::string &str) {
        size_t n = str.size() + 1;
        auto chars = static_cast<char *>(malloc(sizeof(char) * n));
        memcpy(chars, str.c_str(), n);
        return chars;
    }

    void TensorDraw::draw(Tensor *root, const std::string &extension, const std::string &fileName) {
        assert(Error::str_assert(root->graph != nullptr, Error::Message::tensorGraphUninitialized));
        auto tensorGraph = root->graph;
        GVC_t *gvc = gvContext();
        std::string name = "Tensor graph";
        auto nameChars = strToCharPtr(name);
        Agraph_t *gvcGraph = agopen(nameChars, Agdirected, 0);
        Agnode_t *gvcNode1, *gvcNode2;

        // Add nodes
        for (auto &tensor: *tensorGraph) {
            name = std::to_string(tensor->id) + ", ";

            for (auto &op: tensor->ops) {
                name += op2Str[op->opName] + ", ";
            }

            name += tensor->shape.toStr();
            nameChars = strToCharPtr(name);
            agnode(gvcGraph, nameChars, 1);
        }

        // Add edges
        for (auto &tensor: *tensorGraph) {
            name = std::to_string(tensor->id) + ", ";

            for (auto &op: tensor->ops) {
                name += op2Str[op->opName] + ", ";
            }

            name += tensor->shape.toStr();
            nameChars = strToCharPtr(name);
            gvcNode1 = agnode(gvcGraph, nameChars, 0);

            for (auto &neighbor: tensor->edges) {
                name = std::to_string(neighbor->id) + ", ";

                for (auto &op: neighbor->ops) {
                    name += op2Str[op->opName] + ", ";
                }

                name += neighbor->shape.toStr();
                nameChars = strToCharPtr(name);
                gvcNode2 = agnode(gvcGraph, nameChars, 0);
                agedge(gvcGraph, gvcNode1, gvcNode2, 0, 1);
            }
        }

        // Render the graph to a file
        gvLayout(gvc, gvcGraph, "dot");
        gvRenderFilename(gvc, gvcGraph, extension.data(), fileName.data());

        // Clean up
        gvFreeLayout(gvc, gvcGraph);
        agclose(gvcGraph);
        gvFreeContext(gvc);
    }
}
