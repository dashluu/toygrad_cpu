#pragma once

#include <vector>

#include "common.h"

namespace Toygrad::Tensor {
    struct Shape {
    private:
        friend class Tensor;

        Shape() = default;

        void initStridesHelper(std::vector<size_t> &target) const {
            target.resize(view.size());
            target[view.size() - 1] = 1;
            size_t stride = 1;

            for (int i = view.size() - 2; i >= 0; i--) {
                stride *= view[i + 1];
                target[i] = stride;
            }
        }

        void initStrides() {
            initStridesHelper(strides);
        }

        void initVStrides() {
            initStridesHelper(vstrides);
        }

    public:
        size_t offset = 0;
        std::vector<size_t> view;
        std::vector<size_t> strides;
        std::vector<size_t> vstrides;

        Shape(size_t offset, const std::vector<size_t> &view): offset(offset), view(view) {
            initStrides();
        }

        explicit Shape(const std::vector<size_t> &view): Shape(0, view) {
        }

        Shape(const Shape &shape) {
            offset = shape.offset;
            view = shape.view;
            strides = shape.strides;
            vstrides = shape.vstrides;
        }

        void remove(size_t dim) {
            view.erase(view.begin() + dim, view.begin() + dim + 1);
            strides.erase(strides.begin() + dim, strides.begin() + dim + 1);
            vstrides.erase(vstrides.begin() + dim, vstrides.begin() + dim + 1);
        }

        Shape perm(const std::vector<size_t> &shapePerm) const {
            Shape shape(*this);

            for (size_t i = 0; i < shapePerm.size(); i++) {
                shape.view[i] = view[shapePerm[i]];
                shape.strides[i] = strides[shapePerm[i]];
                shape.vstrides[i] = vstrides[shapePerm[i]];
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

            offset = rhs.offset;
            view = rhs.view;
            strides = rhs.strides;
            vstrides = rhs.vstrides;
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

        std::vector<size_t>::const_iterator cbegin() const {
            return view.cbegin();
        }

        std::vector<size_t>::const_iterator cend() const {
            return view.cend();
        }

        std::vector<size_t>::reverse_iterator rbegin() {
            return view.rbegin();
        }

        std::vector<size_t>::reverse_iterator rend() {
            return view.rend();
        }

        std::vector<size_t>::const_reverse_iterator crbegin() const {
            return view.crbegin();
        }

        std::vector<size_t>::const_reverse_iterator crend() const {
            return view.crend();
        }
    };
}
