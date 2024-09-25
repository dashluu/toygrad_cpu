//
// Created by Trung Luu on 9/18/24.
//

#include "nn.h"
#include "tensors/ops.h"

namespace Toygrad::NN {
    Tensor::TensorPtr Module::forward(const std::vector<Tensor::TensorPtr> &x) {
        if (output == nullptr) {
            for (const auto &t: x) {
                t->forward();
                input.push_back(t->copy(false));
            }

            output = F(input);
        } else {
            assert(Error::str_assert(x.size() == input.size(),
                Error::Message::invalidInputSize(x.size(), input.size())));

            for (size_t i = 0; i < x.size(); i++) {
                x[i]->forward();
                input[i] = x[i]->copy(false, input[i]);
            }
        }

        output->forward();
        return output;
    }
}
