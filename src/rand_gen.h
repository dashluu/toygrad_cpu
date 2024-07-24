#pragma once
#include <random>

#include "common.h"

namespace Toygrad::Tensor {
    class RandGen {
        std::mt19937 rng;

        static RandGen &inst() {
            static RandGen randGen;
            return randGen;
        }

    public:
        RandGen() {
            rng = std::mt19937(std::random_device()());
        }

        static int randint(int start, int end) {
            std::uniform_int_distribution<std::mt19937::result_type> dist(start, end);
            return dist(inst().rng);
        }

        static real randn() {
            std::normal_distribution<real> dist(0., 1.);
            return dist(inst().rng);
        }
    };
}
