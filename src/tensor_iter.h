//
// Created by Trung Luu on 7/23/24.
//

#pragma once

#include "tensor.h"

namespace Toygrad::Tensor {
    class TensorIter {
    protected:
        const Tensor *tensor;
        size_t offset = 0;

        explicit TensorIter(const Tensor *tensor): tensor(tensor) {
        }

    public:
        virtual ~TensorIter() = default;

        virtual void start() = 0;

        virtual bool hasNext() = 0;

        virtual void next() = 0;

        virtual real &curr() const = 0;

        virtual size_t count() = 0;
    };

    class DenseIter : public TensorIter {
        size_t elmIdx = 0;

    public:
        explicit DenseIter(const Tensor *tensor): TensorIter(tensor) {
        }

        void start() override {
            elmIdx = tensor->getShape().offset;
        }

        bool hasNext() override {
            return elmIdx < tensor->getShape().offset + tensor->getShape().getSize();
        }

        void next() override {
            elmIdx++;
        }

        real &curr() const override {
            return (*tensor->getVec())[elmIdx];
        }

        size_t count() override {
            return elmIdx + 1;
        }
    };

    class SparseIter : public TensorIter {
        size_t elmIdx = 0;
        int ridx = 0;
        std::vector<size_t> rotator = std::vector<size_t>();
        size_t counter = 0;

    public:
        explicit SparseIter(const Tensor *tensor): TensorIter(tensor) {
            rotator.resize(tensor->getShape().getNumDims());
        }

        void start() override {
            offset = tensor->getShape().offset;
            elmIdx = offset;
            std::ranges::fill(rotator.begin(), rotator.end(), 0);
            ridx = rotator.size() - 1;
            counter = 1;
        }

        bool hasNext() override {
            // TODO: detect out-of-bounds elements
            return counter <= tensor->getShape().getSize();
        }

        void next() override;

        real &curr() const override {
            return (*tensor->getVec())[elmIdx];
        }

        size_t count() override {
            return counter;
        }
    };

    IterPtr initIter(Tensor *tensor);
}
