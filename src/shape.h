#pragma once

#include <vector>
#include "common.h"

namespace Toygrad::Tensor {
    struct Shape {
    private:
        friend class Tensor;
        friend class TensorIndexer;

        Shape() = default;

        void initSt(std::vector<size_t> &st) {
            // Update strides
            st.resize(view.size());
            size_t stride = 1;

            for (int i = view.size() - 2; i >= 0; i--) {
                stride *= view[i + 1];
                st[i] = stride;
            }

            st[view.size() - 1] = 1;
        }

        void initRng() {
            for (size_t &i: view) {
                rng.push_back({0, i, 1});
            }
        }

    public:
        Shape *root = nullptr;
        std::vector<Range> rng;
        size_t offset = 0;
        std::vector<size_t> view;
        // Dynamic strides
        std::vector<size_t> dst;
        // Static strides
        std::vector<size_t> sst;

        Shape(Shape *root, const std::vector<Range> &rng, size_t offset,
              const std::vector<size_t> &view): root(root), rng(rng), offset(offset), view(view) {
            initSt(dst);
            sst = dst;
        }

        Shape(Shape *root, size_t offset, const std::vector<size_t> &view): root(root), offset(offset), view(view) {
            initSt(dst);
            sst = dst;
            initRng();
        }

        Shape(size_t offset, const std::vector<size_t> &view): Shape(this, offset, view) {
        }

        explicit Shape(const std::vector<size_t> &view): Shape(0, view) {
        }

        Shape(const Shape &shape) {
            root = shape.root;
            rng = shape.rng;
            offset = shape.offset;
            view = shape.view;
            dst = shape.dst;
            sst = shape.sst;
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
            rng = rhs.rng;
            offset = rhs.offset;
            view = rhs.view;
            dst = rhs.dst;
            sst = rhs.sst;
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
