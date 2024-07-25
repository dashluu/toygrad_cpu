#pragma once
#include <random>

#include "common.h"

namespace Toygrad::Tensor {
    class RandGen {
        std::mt19937 engine;

        static RandGen &inst() {
            static RandGen randGen;
            return randGen;
        }

    public:
        RandGen() {
            engine = std::mt19937(std::random_device()());
        }

        static int randint(int start, int end) {
            std::uniform_int_distribution<std::mt19937::result_type> dist(start, end);
            return dist(inst().engine);
        }

        static real randn() {
            std::normal_distribution<real> dist(0., 1.);
            return dist(inst().engine);
        }
    };
}
