#include <iostream>
#include "nn/linear.h"
#include "../src/tensors/tensor_draw.h"

using namespace Toygrad::Tensor;
using namespace Toygrad::NN;

class MNISTNN final : public Module {
    std::unique_ptr<Linear> linear;

public:
    MNISTNN() {
        linear = std::make_unique<Linear>(784, 10);
    }

    TensorPtr F(const std::vector<TensorPtr> &x) override {
        auto t1 = linear->F(x);
        auto t2 = t1->sum();
        return t2;
    }
};

int main() {
    auto t1 = MNISTNN();
    TensorPtr output;
    constexpr size_t numBatches = 5;
    std::cout << "Progress: ";

    for (size_t i = 0; i < numBatches; i++) {
        constexpr size_t batchSize = 64;
        auto t2 = Tensor::randn({batchSize, 784});
        output = t1.forward({t2});
        output->backward();

        if (i % 25 == 0) {
            std::cout << "=" << std::flush;
        }
    }

    std::cout << std::endl;
    TensorDraw tensorDraw;
    tensorDraw.draw(output.get(), "png", "output.png");
    return 0;
}
