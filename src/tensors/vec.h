#pragma once
#include <iostream>
#include "common.h"

namespace Toygrad::Tensor {
    struct Vec {
        size_t size;
        std::unique_ptr<real[]> buff;

        explicit Vec(size_t size) : size(size) {
            buff = std::make_unique<real[]>(size);
        }

        Vec(size_t size, real c) : Vec(size) {
            std::fill_n(buff.get(), size, c);
        }

        Vec(const Vec &vec) {
            size = vec.size;
            buff = std::make_unique<real[]>(size);
            std::ranges::copy(vec.buff.get(), vec.buff.get() + size, buff.get());
        }

        ~Vec() = default;

        real &operator[](size_t idx) const {
            return buff[idx];
        }

        friend std::ostream &operator<<(std::ostream &stream, const Vec &vec);
    };
}
