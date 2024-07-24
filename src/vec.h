#pragma once
#include <iostream>
#include "common.h"

namespace Toygrad::Tensor {
    class Tensor;

    struct Vec {
    private:
        size_t refCount = 0;
        friend class Tensor;

    public:
        size_t size;
        real *buff;

        explicit Vec(size_t size) : size(size) {
            buff = new real[size];
        }

        Vec(size_t size, real c) : Vec(size) {
            std::fill_n(buff, size, c);
        }

        Vec(const Vec &vec) {
            size = vec.size;
            buff = new real[size];
            std::ranges::copy(vec.buff, vec.buff + size, buff);
        }

        ~Vec() {
            delete buff;
        }

        real &operator[](size_t idx) const {
            return buff[idx];
        }

        friend std::ostream &operator<<(std::ostream &stream, const Vec &vec);
    };
}
