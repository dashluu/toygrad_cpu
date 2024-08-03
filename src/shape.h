#pragma once

#include <vector>

#include "common.h"

namespace Toygrad::Tensor {
    struct Shape {
    private:
        friend class Tensor;

        Shape() = default;

        void defStrides() {
            // Update strides
            strides.resize(view.size());
            size_t stride = 1;

            for (int i = view.size() - 2; i >= 0; i--) {
                stride *= view[i + 1];
                strides[i] = stride;
            }

            strides[view.size() - 1] = 1;
        }

        void defRanges() {
            ranges.resize(view.size());

            for (size_t i = 0; i < view.size(); i++) {
                ranges[i] = {0, view[i], 1};
            }
        }

    public:
        std::vector<Range> ranges;
        size_t offset = 0;
        std::vector<size_t> view;
        std::vector<size_t> strides;

        Shape(size_t offset, const std::vector<size_t> &view): offset(offset), view(view) {
            defStrides();
            defRanges();
        }

        explicit Shape(const std::vector<size_t> &view): Shape(0, view) {
        }

        Shape(const Shape &shape) {
            ranges = shape.ranges;
            offset = shape.offset;
            view = shape.view;
            strides = shape.strides;
        }

        Shape perm(const std::vector<size_t> &shapePerm) const {
            Shape shape(*this);

            for (size_t i = 0; i < shapePerm.size(); i++) {
                shape.ranges[i] = ranges[shapePerm[i]];
                shape.view[i] = view[shapePerm[i]];
                shape.strides[i] = strides[shapePerm[i]];
            }

            return shape;
        }

        bool operator==(const Shape &rhs) const {
            return view == rhs.view;
        }

        bool operator!=(const Shape &rhs) const {
            return !(*this == rhs);
        }

        Shape &operator=(const Shape &rhs) {
            if (this == &rhs) {
                return *this;
            }

            ranges = rhs.ranges;
            offset = rhs.offset;
            view = rhs.view;
            strides = rhs.strides;
            return *this;
        }

        size_t getNumDims() const {
            return view.size();
        }

        size_t getSize() const {
            size_t s = 1;

            for (size_t dim: view) {
                s *= dim;
            }

            return s;
        }

        size_t &operator[](size_t idx) {
            return view[idx];
        }

        std::vector<size_t>::iterator begin() {
            return view.begin();
        }

        std::vector<size_t>::iterator end() {
            return view.end();
        }
    };
}
