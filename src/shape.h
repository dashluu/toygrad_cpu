#pragma once

#include <vector>
#include "common.h"

namespace Toygrad::Tensor {
    class Tensor;
    class TensorIndexer;

    struct Shape {
    private:
        friend class Tensor;
        friend class TensorIndexer;

        explicit Shape(size_t numDims) {
            view.resize(numDims);
            strides.resize(numDims);
        }

        Shape() = default;

        void initStrides() {
            // Update strides
            strides.resize(view.size());
            size_t stride = 1;

            for (int i = view.size() - 2; i >= 0; i--) {
                stride *= view[i + 1];
                strides[i] = stride;
            }

            strides[view.size() - 1] = 1;
        }

        void initRanges() {
            for (size_t &i: view) {
                ranges.push_back({0, i, 1});
            }
        }

    public:
        Shape *root = nullptr;
        std::vector<Range> ranges;
        size_t offset = 0;
        std::vector<size_t> view;
        std::vector<size_t> strides;

        Shape(Shape *root, const std::vector<Range> &ranges, size_t offset,
              const std::vector<size_t> &view): root(root), ranges(ranges), offset(offset), view(view) {
            initStrides();
        }

        Shape(Shape *root, size_t offset, const std::vector<size_t> &view): root(root), offset(offset), view(view) {
            initStrides();
            initRanges();
        }

        Shape(size_t offset, const std::vector<size_t> &view): Shape(this, offset, view) {
        }

        explicit Shape(const std::vector<size_t> &view): Shape(0, view) {
        }

        Shape(const Shape &shape) {
            root = shape.root;
            ranges = shape.ranges;
            offset = shape.offset;
            view = shape.view;
            strides = shape.strides;
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

            root = rhs.root;
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
