#include <iostream>

#include <ranges>
#include "tensor.h"
#include "ops.h"
#include "tensor_iter.h"
#include "str_assert.h"
#include "tensor_graph.h"

namespace Toygrad::Tensor {
    size_t Tensor::idCounter = 0;

    Tensor::Tensor() {
        id = idCounter++;
    }

    Tensor::Tensor(const Shape &shape, bool initStrides) : Tensor() {
        this->shape = initStrides ? Shape(0, shape.view) : shape;
    }

    Tensor::~Tensor() {
        std::cout << "Destroyed tensor " << id << "..." << std::endl;
        delete graph;
        clearOps();
    }

    std::ostream &operator<<(std::ostream &stream, const Tensor &tensor) {
        IterPtr iter = initConstIter(&tensor);
        std::vector<size_t> sizePerDim = tensor.shape.getSizePerDim();
        iter->start();
        bool flag = iter->hasNext();

        for (size_t i = 0; i < tensor.shape.getNumDims(); i++) {
            stream << "[";
        }

        if (!flag) {
            for (size_t i = 0; i < tensor.shape.getNumDims(); i++) {
                stream << "]";
            }

            return stream;
        }

        do {
            stream << iter->curr();
            size_t close = 0;

            for (int i = sizePerDim.size() - 1; i >= 0; i--) {
                if (iter->count() % sizePerDim[i] == 0) {
                    stream << "]";
                    close++;
                }
            }

            iter->next();
            flag = iter->hasNext();

            if (flag) {
                stream << ", ";

                if (close > 0) {
                    stream << std::endl;
                }

                for (size_t i = 0; i < close; i++) {
                    stream << "[";
                }
            }
        } while (flag);

        return stream;
    }

    void Tensor::clearOps() {
        for (const auto &op: ops) {
            delete op;
        }

        ops.clear();
    }

    TensorPtr Tensor::index(const std::vector<size_t> &indices) {
        // Multidimensional tensor
        assert(str_assert(shape.getNumDims() >= indices.size(), AssertMessage::indexMultidimsOnly));

        // Index must stay within bounds
        for (size_t i = 0; i < indices.size(); i++) {
            assert(str_assert(indices[i] < shape[i], AssertMessage::indexOutOfBounds));
        }

        auto outShape = shape;

        for (size_t idx: indices) {
            outShape.offset += idx * outShape.strides[0];
            outShape.remove(0);
        }

        auto ranges = std::vector<Range>();

        for (size_t i = indices.size(); i < shape.getNumDims(); i++) {
            ranges.push_back({0, shape[i], 1});
        }

        auto outTensor = alias(outShape);
        return outTensor->index(ranges);
    }

    TensorPtr Tensor::index(const std::vector<Range> &ranges) {
        Shape outShape;
        outShape.offset = shape.offset;

        for (size_t i = 0; i < ranges.size(); i++) {
            outShape.offset += ranges[i].beg * shape.strides[i];
        }

        for (size_t i = 0; i < ranges.size(); i++) {
            size_t dim = ceil(static_cast<real>(ranges[i].end - ranges[i].beg) / ranges[i].step);
            outShape.view.push_back(dim);
            outShape.strides.push_back(shape.strides[i] * ranges[i].step);
        }

        return alias(outShape);
    }

    bool Tensor::isContiguous() const {
        return shape.strides == shape.getContiguousStrides();
    }

    bool Tensor::isBroadcastableTo(const Shape &target) const {
        if (shape == target) {
            return true;
        }

        if (shape.getNumDims() > target.getNumDims()) {
            return false;
        }

        for (auto shapeIter = shape.crbegin(), targetIter = target.crbegin();
             shapeIter != shape.crend() && targetIter != target.crend();
             ++shapeIter, ++targetIter) {
            if (*shapeIter != 1 && *shapeIter != *targetIter) {
                return false;
            }
        }

        return true;
    }

    TensorPtr Tensor::broadcastTo(const Shape &target) {
        if (shape == target) {
            return getThis();
        }

        assert(str_assert(isBroadcastableTo(target), AssertMessage::notBroadcastable));
        Shape outShape;
        outShape.offset = shape.offset;
        outShape.view = shape.view;
        size_t dimsToAdd = target.getNumDims() - outShape.getNumDims();

        for (size_t i = 0; i < dimsToAdd; i++) {
            outShape.view.insert(outShape.view.begin(), 1);
        }

        outShape.initStrides();

        for (int i = target.getNumDims() - 1; i >= 0; i--) {
            if (outShape.view[i] < target.view[i]) {
                // outShape.view[i] == 1
                outShape.view[i] = target.view[i];
                outShape.strides[i] = 0;
            }
        }

        return alias(outShape);
    }

    TensorPtr Tensor::alias(const Shape &target) {
        auto outTensor = initTensor(target, false);
        outTensor->ops.push_back(new AliasOp(getThis(), outTensor.get()));
        return outTensor;
    }

    TensorPtr Tensor::alias() {
        return alias(shape);
    }

    TensorPtr Tensor::diffAlias() {
        auto outTensor = initTensor(shape, false);
        outTensor->ops.push_back(new DiffAliasOp(getThis(), outTensor.get()));
        return outTensor;
    }

    TensorPtr Tensor::copy() {
        auto outTensor = initTensor(shape);
        outTensor->ops.push_back(new CopyOp(getThis(), outTensor.get()));
        return outTensor;
    }

    bool Tensor::isSqueezable(int64_t dim) const {
        if (!isDimValid(dim)) {
            return false;
        }

        if (dim != -1) {
            return shape.view[dim] == 1 ? shape.getNumDims() > 1 : true;
        }

        size_t dimsToSqueeze = 0;

        for (auto shapeIter = shape.cbegin(); shapeIter != shape.cend(); ++shapeIter) {
            if (*shapeIter == 1) {
                dimsToSqueeze++;
            }
        }

        return shape.getNumDims() > dimsToSqueeze;
    }

    TensorPtr Tensor::squeeze(int64_t dim) {
        Shape outShape;

        if (dim != -1) {
            assert(str_assert(isDimValid(dim), AssertMessage::invalidDim));
            outShape = shape;

            if (outShape[dim] == 1) {
                outShape.remove(dim);
            }
        } else {
            assert(str_assert(isSqueezable(dim), AssertMessage::notSqueezable));
            outShape = shape;
            int i = 0;

            while (i < outShape.getNumDims()) {
                if (outShape[i] == 1) {
                    outShape.remove(i);
                } else {
                    i++;
                }
            }
        }

        return alias(outShape);
    }

    TensorPtr Tensor::unsqueeze(int64_t dim) {
        assert(str_assert(isDimValid(dim), AssertMessage::invalidDim));
        Shape outShape = shape;

        if (dim == -1) {
            outShape.view.push_back(1);
            outShape.strides.push_back(1);
        } else {
            outShape.view.insert(outShape.view.begin() + dim, 1);
            size_t stride = outShape.view[dim] * outShape.strides[dim];
            outShape.strides.insert(outShape.strides.begin() + dim, stride);
        }

        return alias(outShape);
    }

    TensorPtr Tensor::at(size_t idx) {
        return index({idx});
    }

    TensorPtr Tensor::at(const std::vector<size_t> &indices) {
        return index(indices);
    }

    TensorPtr Tensor::at(const std::vector<Range> &ranges) {
        assert(str_assert(shape.getNumDims() >= ranges.size(), AssertMessage::indexMultidimsOnly));
        std::vector<Range> newRanges = ranges;

        // Turn invalid ranges into valid ones
        for (size_t i = 0; i < newRanges.size(); i++) {
            if (newRanges[i].beg >= shape.view[i]) {
                newRanges[i].beg = 0;
                newRanges[i].end = 0;
            } else if (newRanges[i].end > shape.view[i]) {
                newRanges[i].end = shape.view[i];
            }
        }

        return index(newRanges);
    }

    TensorPtr Tensor::arange(const Shape &shape, real start, real step) {
        auto outTensor = initTensor(shape);
        outTensor->ops.push_back(new ArangeOp(outTensor.get(), start, step));
        return outTensor;
    }

    TensorPtr Tensor::randint(const Shape &shape, int64_t min, int64_t max) {
        auto outTensor = initTensor(shape);
        outTensor->ops.push_back(new RandintOp(outTensor.get(), min, max));
        return outTensor;
    }

    TensorPtr Tensor::randn(const Shape &shape) {
        auto outTensor = initTensor(shape);
        outTensor->ops.push_back(new RandnOp(outTensor.get()));
        return outTensor;
    }

    TensorPtr Tensor::fromConst(const Shape &shape, real c) {
        auto outTensor = initTensor(shape);
        outTensor->ops.push_back(new ConstOp(outTensor.get(), c));
        return outTensor;
    }

    TensorPtr Tensor::fromArr(const Shape &shape, const real *data) {
        auto outTensor = initTensor(shape);
        outTensor->ops.push_back(new FromArrOp(outTensor.get(), data));
        return outTensor;
    }

    TensorPtr Tensor::operator[](size_t idx) {
        auto outTensor = at(idx);
        return outTensor;
    }

    TensorPtr Tensor::add(Tensor &rhs) {
        assert(str_assert(rhs.isBroadcastableTo(shape), AssertMessage::notBroadcastable));
        auto broadcastedRhs = rhs.broadcastTo(shape);
        auto outTensor = initTensor(shape);
        outTensor->ops.push_back(new AddOp(getThis(), broadcastedRhs, outTensor.get()));
        return outTensor;
    }

    TensorPtr Tensor::add(real c) {
        return add(fromConst(shape, c));
    }

    TensorPtr Tensor::sub(Tensor &rhs) {
        assert(str_assert(rhs.isBroadcastableTo(shape), AssertMessage::notBroadcastable));
        auto broadcastedRhs = rhs.broadcastTo(shape);
        auto outTensor = initTensor(shape);
        outTensor->ops.push_back(new SubOp(getThis(), broadcastedRhs, outTensor.get()));
        return outTensor;
    }

    TensorPtr Tensor::sub(real c) {
        return sub(fromConst(shape, c));
    }

    TensorPtr Tensor::mul(Tensor &rhs) {
        assert(str_assert(rhs.isBroadcastableTo(shape), AssertMessage::notBroadcastable));
        auto broadcastedRhs = rhs.broadcastTo(shape);
        auto outTensor = initTensor(shape);
        outTensor->ops.push_back(new MulOp(getThis(), broadcastedRhs, outTensor.get()));
        return outTensor;
    }

    TensorPtr Tensor::mul(real c) {
        return mul(fromConst(shape, c));
    }

    TensorPtr Tensor::div(Tensor &rhs) {
        assert(str_assert(rhs.isBroadcastableTo(shape), AssertMessage::notBroadcastable));
        auto broadcastedRhs = rhs.broadcastTo(shape);
        auto outTensor = initTensor(shape);
        outTensor->ops.push_back(new DivOp(getThis(), broadcastedRhs, outTensor.get()));
        return outTensor;
    }

    TensorPtr Tensor::div(real c) {
        return div(fromConst(shape, c));
    }

    TensorPtr Tensor::pow(real c) {
        auto outTensor = initTensor(shape);
        outTensor->ops.push_back(new PowOp(getThis(), outTensor.get(), c));
        return outTensor;
    }

    TensorPtr Tensor::log() {
        auto outTensor = initTensor(shape);
        outTensor->ops.push_back(new LogOp(getThis(), outTensor.get()));
        return outTensor;
    }

    TensorPtr Tensor::sin() {
        auto outTensor = initTensor(shape);
        outTensor->ops.push_back(new SinOp(getThis(), outTensor.get()));
        return outTensor;
    }

    TensorPtr Tensor::cos() {
        auto outTensor = initTensor(shape);
        outTensor->ops.push_back(new CosOp(getThis(), outTensor.get()));
        return outTensor;
    }

    TensorPtr Tensor::exp() {
        auto outTensor = initTensor(shape);
        outTensor->ops.push_back(new ExpOp(getThis(), outTensor.get()));
        return outTensor;
    }

    TensorPtr Tensor::recip(real c) {
        auto outTensor = initTensor(shape);
        outTensor->ops.push_back(new RecipOp(getThis(), outTensor.get(), c));
        return outTensor;
    }

    TensorPtr Tensor::sq() {
        auto outTensor = initTensor(shape);
        outTensor->ops.push_back(new SqOp(getThis(), outTensor.get()));
        return outTensor;
    }

    TensorPtr Tensor::sqrt() {
        auto outTensor = initTensor(shape);
        outTensor->ops.push_back(new SqrtOp(getThis(), outTensor.get()));
        return outTensor;
    }

    TensorPtr Tensor::neg() {
        auto outTensor = initTensor(shape);
        outTensor->ops.push_back(new NegOp(getThis(), outTensor.get()));
        return outTensor;
    }

    TensorPtr Tensor::eq(Tensor &rhs) {
        assert(str_assert(rhs.isBroadcastableTo(shape), AssertMessage::notBroadcastable));
        auto broadcastedRhs = rhs.broadcastTo(shape);
        auto outTensor = initTensor(shape);
        outTensor->ops.push_back(new EqOp(getThis(), broadcastedRhs, outTensor.get()));
        return outTensor;
    }

    TensorPtr Tensor::eq(real c) {
        return eq(fromConst(shape, c));
    }

    bool Tensor::operator==(const Tensor &rhs) const {
        if (shape != rhs.shape) {
            return false;
        }

        // TODO: convert this to initIter
        IterPtr lhsIter = initConstIter(this);
        IterPtr rhsIter = initConstIter(&rhs);

        for (lhsIter->start(), rhsIter->start(); lhsIter->hasNext(); lhsIter->next(), rhsIter->next()) {
            if (lhsIter->curr() != rhsIter->curr()) {
                return false;
            }
        }

        return true;
    }

    TensorPtr Tensor::neq(Tensor &rhs) {
        assert(str_assert(rhs.isBroadcastableTo(shape), AssertMessage::notBroadcastable));
        auto broadcastedRhs = rhs.broadcastTo(shape);
        auto outTensor = initTensor(shape);
        outTensor->ops.push_back(new NeqOp(getThis(), broadcastedRhs, outTensor.get()));
        return outTensor;
    }

    TensorPtr Tensor::neq(real c) {
        return neq(fromConst(shape, c));
    }

    bool Tensor::operator!=(const Tensor &rhs) const {
        return !(*this == rhs);
    }

    TensorPtr Tensor::lt(Tensor &rhs) {
        assert(str_assert(rhs.isBroadcastableTo(shape), AssertMessage::notBroadcastable));
        auto broadcastedRhs = rhs.broadcastTo(shape);
        auto outTensor = initTensor(shape);
        outTensor->ops.push_back(new LessOp(getThis(), broadcastedRhs, outTensor.get()));
        return outTensor;
    }

    TensorPtr Tensor::lt(real c) {
        return lt(fromConst(shape, c));
    }

    TensorPtr Tensor::gt(Tensor &rhs) {
        assert(str_assert(rhs.isBroadcastableTo(shape), AssertMessage::notBroadcastable));
        auto broadcastedRhs = rhs.broadcastTo(shape);
        auto outTensor = initTensor(shape);
        outTensor->ops.push_back(new GreaterOp(getThis(), broadcastedRhs, outTensor.get()));
        return outTensor;
    }

    TensorPtr Tensor::gt(real c) {
        return gt(fromConst(shape, c));
    }

    TensorPtr Tensor::leq(Tensor &rhs) {
        assert(str_assert(rhs.isBroadcastableTo(shape), AssertMessage::notBroadcastable));
        auto broadcastedRhs = rhs.broadcastTo(shape);
        auto outTensor = initTensor(shape);
        outTensor->ops.push_back(new LeqOp(getThis(), broadcastedRhs, outTensor.get()));
        return outTensor;
    }

    TensorPtr Tensor::leq(real c) {
        return leq(fromConst(shape, c));
    }

    TensorPtr Tensor::geq(Tensor &rhs) {
        assert(str_assert(rhs.isBroadcastableTo(shape), AssertMessage::notBroadcastable));
        auto broadcastedRhs = rhs.broadcastTo(shape);
        auto outTensor = initTensor(shape);
        outTensor->ops.push_back(new GeqOp(getThis(), broadcastedRhs, outTensor.get()));
        return outTensor;
    }

    TensorPtr Tensor::geq(real c) {
        return geq(fromConst(shape, c));
    }

    TensorPtr Tensor::addAssign(Tensor &rhs) {
        assert(str_assert(rhs.isBroadcastableTo(shape), AssertMessage::notBroadcastable));
        auto broadcastedRhs = rhs.broadcastTo(shape);
        ops.push_back(new AddAssignOp(broadcastedRhs, this));
        return getThis();
    }

    TensorPtr Tensor::addAssign(real c) {
        return addAssign(fromConst(shape, c));
    }

    TensorPtr Tensor::subAssign(Tensor &rhs) {
        assert(str_assert(rhs.isBroadcastableTo(shape), AssertMessage::notBroadcastable));
        auto broadcastedRhs = rhs.broadcastTo(shape);
        ops.push_back(new SubAssignOp(broadcastedRhs, this));
        return getThis();
    }

    TensorPtr Tensor::subAssign(real c) {
        return subAssign(fromConst(shape, c));
    }

    TensorPtr Tensor::mulAssign(Tensor &rhs) {
        assert(str_assert(rhs.isBroadcastableTo(shape), AssertMessage::notBroadcastable));
        auto broadcastedRhs = rhs.broadcastTo(shape);
        ops.push_back(new MulAssignOp(broadcastedRhs, this));
        return getThis();
    }

    TensorPtr Tensor::mulAssign(real c) {
        return mulAssign(fromConst(shape, c));
    }

    TensorPtr Tensor::divAssign(Tensor &rhs) {
        assert(str_assert(rhs.isBroadcastableTo(shape), AssertMessage::notBroadcastable));
        auto broadcastedRhs = rhs.broadcastTo(shape);
        ops.push_back(new DivAssignOp(broadcastedRhs, this));
        return getThis();
    }

    TensorPtr Tensor::divAssign(real c) {
        return divAssign(fromConst(shape, c));
    }

    TensorPtr Tensor::relu() {
        auto outTensor = initTensor(shape);
        outTensor->ops.push_back(new ReluOp(getThis(), outTensor.get()));
        return outTensor;
    }

    TensorPtr Tensor::sigmoid() {
        auto outTensor = initTensor(shape);
        outTensor->ops.push_back(new SigmoidOp(getThis(), outTensor.get()));
        return outTensor;
    }

    TensorPtr Tensor::softmax(int64_t dim) {
        assert(str_assert(isDimValid(dim), AssertMessage::invalidDim));
        TensorPtr outTensor;

        if (dim == -1) {
            auto maxTensor = max();
            auto subTensor = sub(maxTensor);
            auto expTensor = subTensor->exp();
            auto sumTensor = expTensor->sum();
            outTensor = expTensor->div(sumTensor);
        } else {
            std::vector<size_t> shapePerm(shape.getNumDims());
            std::iota(shapePerm.begin(), shapePerm.end(), 0);
            shapePerm.erase(shapePerm.begin() + dim);
            shapePerm.push_back(dim);
            // Permutes operand
            Shape permShape = shape.perm(shapePerm);
            auto permTensor = perm(permShape);
            // std::cout << std::endl << "Perm:" << std::endl << *permTensor << std::endl;
            // Compute softmax
            auto maxTensor = permTensor->max(shape.getNumDims() - 1)->unsqueeze();
            // std::cout << std::endl << "Max:" << std::endl << *maxTensor << std::endl;
            auto subTensor = permTensor->sub(maxTensor);
            // std::cout << std::endl << "Sub:" << std::endl << *subTensor << std::endl;
            auto expTensor = subTensor->exp();
            // std::cout << std::endl << "Exp:" << std::endl << *expTensor << std::endl;
            auto sumTensor = expTensor->sum(shape.getNumDims() - 1)->unsqueeze();
            // std::cout << std::endl << "Sum:" << std::endl << *sumTensor << std::endl;
            auto softmaxTensor = expTensor->div(sumTensor);
            // std::cout << std::endl << "Softmax:" << std::endl << *softmaxTensor << std::endl;
            // Permute softmax
            std::iota(shapePerm.begin(), shapePerm.end(), 0);
            size_t lastDim = shapePerm[shapePerm.size() - 1];
            shapePerm.erase(shapePerm.end() - 1);
            shapePerm.insert(shapePerm.begin() + dim, lastDim);
            outTensor = softmaxTensor->perm(shapePerm);
        }

        return outTensor;
    }

    TensorPtr Tensor::matmul(Tensor &rhs) {
        assert(str_assert(shape.getNumDims() == rhs.shape.getNumDims(), AssertMessage::shapesMismatched));
        assert(str_assert(shape.getNumDims() >= 2, AssertMessage::matmulOnLessThan2d));
        assert(str_assert(std::equal(shape.begin(), shape.end() - 2,
                rhs.shape.begin(), rhs.shape.end() - 2),
            AssertMessage::shapesMismatched));
        size_t numDims = shape.getNumDims();
        assert(str_assert(shape[numDims - 1] == rhs.shape[numDims - 2], AssertMessage::shapesMismatched));
        Shape outShape = shape;
        // Shape of ...x H1 x W1 matmul ...x W1 x H2 == ...x H1 x H2
        outShape[numDims - 1] = rhs.shape[numDims - 1];
        // Permutes rhs's last two dimensions
        auto tranposedRhs = rhs.T(numDims - 2);
        // Do matrix multiplication on the last 2 dimensions
        auto outTensor = initTensor(outShape);
        outTensor->ops.push_back(new MatmulOp(getThis(), tranposedRhs, outTensor.get()));
        return outTensor;
    }

    TensorPtr Tensor::reshape(const Shape &target) {
        assert(str_assert(target.getSize() == shape.getSize(), AssertMessage::shapesMismatched));
        TensorPtr outTensor;

        if (isContiguous()) {
            outTensor = initTensor(target, false);
            outTensor->shape.offset = shape.offset;
            outTensor->ops.push_back(new AliasOp(getThis(), outTensor.get()));
        } else {
            outTensor = initTensor(target);
            outTensor->ops.push_back(new CopyOp(getThis(), outTensor.get()));
        }

        return outTensor;
    }

    TensorPtr Tensor::sum(int64_t dim) {
        assert(str_assert(isDimValid(dim), AssertMessage::invalidDim));
        TensorPtr outTensor;

        if (dim == -1) {
            outTensor = initTensor(Shape({1}));
            outTensor->ops.push_back(new SumOp(getThis(), outTensor.get(), dim));
        } else {
            Shape outShape = shape;
            // Remove the dimension in which the sum is computed in from the output shape
            outShape.view.erase(outShape.view.begin() + dim);
            // Move the dimension in which the sum is computed in to the back of the input shape
            std::vector<size_t> shapePerm(shape.getNumDims());
            std::iota(shapePerm.begin(), shapePerm.end(), 0);
            shapePerm.erase(shapePerm.begin() + dim);
            shapePerm.push_back(dim);
            // Compute sum
            auto permTensor = perm(shapePerm);
            outTensor = initTensor(outShape);
            outTensor->ops.push_back(new SumOp(permTensor, outTensor.get(), dim));
        }

        return outTensor;
    }

    TensorPtr Tensor::max(int64_t dim) {
        assert(str_assert(isDimValid(dim), AssertMessage::invalidDim));
        TensorPtr outTensor;

        if (dim == -1) {
            outTensor = initTensor(Shape({1}));
            outTensor->ops.push_back(new MaxOp(getThis(), outTensor.get(), dim));
        } else {
            Shape outShape = shape;
            // Remove the dimension in which the max is computed in from the output shape
            outShape.view.erase(outShape.view.begin() + dim);
            // Move the dimension in which the max is computed in to the back of the input shape
            std::vector<size_t> shapePerm(shape.getNumDims());
            std::iota(shapePerm.begin(), shapePerm.end(), 0);
            shapePerm.erase(shapePerm.begin() + dim);
            shapePerm.push_back(dim);
            // Compute max
            auto permTensor = perm(shapePerm);
            outTensor = initTensor(outShape);
            outTensor->ops.push_back(new MaxOp(permTensor, outTensor.get(), dim));
        }

        return outTensor;
    }

    TensorPtr Tensor::min(int64_t dim) {
        assert(str_assert(isDimValid(dim), AssertMessage::invalidDim));
        TensorPtr outTensor;

        if (dim == -1) {
            outTensor = initTensor(Shape({1}));
            outTensor->ops.push_back(new MinOp(getThis(), outTensor.get(), dim));
        } else {
            Shape outShape = shape;
            // Remove the dimension in which the min is computed in from the output shape
            outShape.view.erase(outShape.view.begin() + dim);
            // Move the dimension in which the min is computed in to the back of the input shape
            std::vector<size_t> shapePerm(shape.getNumDims());
            std::iota(shapePerm.begin(), shapePerm.end(), 0);
            shapePerm.erase(shapePerm.begin() + dim);
            shapePerm.push_back(dim);
            // Compute min
            auto permTensor = perm(shapePerm);
            outTensor = initTensor(outShape);
            outTensor->ops.push_back(new MinOp(permTensor, outTensor.get(), dim));
        }

        return outTensor;
    }

    TensorPtr Tensor::perm(const std::vector<size_t> &shapePerm) {
        assert(str_assert(shapePerm.size() == shape.getNumDims(), AssertMessage::invalidShapePerm));
        std::vector<bool> flags(shape.getNumDims());

        for (size_t i: shapePerm) {
            assert(str_assert(i < shape.getNumDims(), AssertMessage::invalidShapePerm));
            flags[i] = true;
        }

        for (bool i: flags) {
            assert(str_assert(i, AssertMessage::invalidShapePerm));
        }

        Shape permShape = shape.perm(shapePerm);
        auto outTensor = initTensor(permShape, false);
        outTensor->ops.push_back(new PermOp(getThis(), outTensor.get()));
        return outTensor;
    }

    TensorPtr Tensor::perm(const Shape &target) {
        auto outTensor = initTensor(target, false);
        outTensor->ops.push_back(new PermOp(getThis(), outTensor.get()));
        return outTensor;
    }

    TensorPtr Tensor::T(size_t startDim) {
        assert(str_assert(startDim < shape.getNumDims(), AssertMessage::invalidDim));
        std::vector<size_t> shapePerm(shape.getNumDims());
        std::iota(shapePerm.begin(), shapePerm.end(), 0);
        std::reverse(shapePerm.begin() + startDim, shapePerm.end());
        return perm(shapePerm);
    }

    bool Tensor::isEmpty() const {
        IterPtr iter = initConstIter(this);
        iter->start();
        return !iter->hasNext();
    }

    void Tensor::forward() {
        if (graph == nullptr) {
            graph = new TensorGraph(getThis());
        }

        graph->forward();
    }

    void Tensor::backward() {
        if (graph == nullptr) {
            graph = new TensorGraph(getThis());
        }

        graph->backward();
    }
}
