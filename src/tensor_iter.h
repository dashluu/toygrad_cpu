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

        virtual void save() = 0;

        virtual void restore() = 0;
    };

    class DenseIter final : public TensorIter {
        struct State {
            size_t elmIdx = 0;

            State() = default;
        };

        State state = State();
        std::vector<State> saved = std::vector<State>();

    public:
        explicit DenseIter(const Tensor *tensor): TensorIter(tensor) {
        }

        void start() override {
            offset = tensor->getShape().offset;
            state.elmIdx = offset;
        }

        bool hasNext() override {
            return state.elmIdx < tensor->getShape().offset + tensor->getShape().getSize();
        }

        void next() override {
            state.elmIdx++;
        }

        real &curr() const override {
            return (*tensor->getVec())[state.elmIdx];
        }

        size_t count() override {
            return state.elmIdx - offset + 1;
        }

        void save() override {
            saved.push_back(state);
        }

        void restore() override {
            if (saved.empty()) return;
            state = saved.back();
            saved.pop_back();
        }
    };

    class SparseIter : public TensorIter {
        struct State {
            size_t elmIdx = 0;
            size_t ridx = 0;
            std::vector<size_t> rotator = std::vector<size_t>();
            size_t counter = 0;

            State() = default;
        };

        State state = State();
        std::vector<State> saved = std::vector<State>();

    public:
        explicit SparseIter(const Tensor *tensor): TensorIter(tensor) {
            state.rotator.resize(tensor->getShape().getNumDims());
        }

        void start() override {
            offset = tensor->getShape().offset;
            state.elmIdx = offset;
            std::ranges::fill(state.rotator.begin(), state.rotator.end(), 0);
            state.ridx = state.rotator.size() - 1;
            state.counter = 1;
        }

        bool hasNext() override {
            return state.counter <= tensor->getShape().getSize();
        }

        void next() override;

        real &curr() const override {
            return (*tensor->getVec())[state.elmIdx];
        }

        size_t count() override {
            return state.counter;
        }

        void save() override {
            saved.push_back(state);
        }

        void restore() override {
            if (saved.empty()) return;
            state = saved.back();
            saved.pop_back();
        }
    };

    IterPtr initIter(Tensor *tensor);

    IterPtr initConstIter(const Tensor *tensor);
}
