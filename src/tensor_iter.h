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
        int nIdx = 0;
        std::vector<size_t> nIndices = std::vector<size_t>();
        size_t counter = 0;

    public:
        explicit SparseIter(const Tensor *tensor): TensorIter(tensor) {
        }

        void start() override {
            offset = tensor->getShape().offset;
            elmIdx = offset;
            nIndices.resize(tensor->getShape().getNumDims());
            nIdx = nIndices.size() - 1;
            counter = 0;
        }

        bool hasNext() override {
            // TODO: detect out-of-bounds elements
            return counter < tensor->getShape().getSize();
        }

        void next() override;

        real &curr() const override {
            return (*tensor->getVec())[elmIdx];
        }

        size_t count() override {
            return counter + 1;
        }
    };

    IterPtr initIter(Tensor *tensor);
}
