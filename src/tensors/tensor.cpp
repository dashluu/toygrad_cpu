#include <iostream>
#include <sstream>
#include <ranges>
#include "tensor.h"
#include "ops.h"
#include "tensor_iter.h"
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
        // std::cout << "Destroyed tensor " << id << "..." << std::endl;
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
        for (auto &op: ops) {
            delete op;
        }

        ops.clear();
    }

    void Tensor::realizeOp(Op *op, bool lazy) {
        if (!lazy) {
            op->forward();
            delete op;
        }
    }

    TensorPtr Tensor::index(const std::vector<size_t> &indices, bool lazy, TensorPtr outTensor) {
        // Multidimensional tensor
        assert(Error::str_assert(shape.getNumDims() >= indices.size(), Error::Message::indexMultidimsOnly));

        // Index must stay within bounds
        for (size_t i = 0; i < indices.size(); i++) {
            assert(Error::str_assert(indices[i] < shape[i], Error::Message::indexOutOfBounds));
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

        outTensor = alias(outShape, lazy, nullptr);
        return outTensor->index(ranges, lazy, outTensor);
    }

    TensorPtr Tensor::index(const std::vector<Range> &ranges, bool lazy, TensorPtr outTensor) {
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

        return alias(outShape, lazy, std::move(outTensor));
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

    TensorPtr Tensor::broadcastTo(const Shape &target, bool lazy, TensorPtr outTensor) {
        if (shape == target) {
            return getThis();
        }

        assert(Error::str_assert(isBroadcastableTo(target), Error::Message::notBroadcastable(shape, target)));
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

        return alias(outShape, lazy, std::move(outTensor));
    }

    TensorPtr Tensor::alias(const Shape &target, bool lazy, TensorPtr outTensor) {
        outTensor = initTensor(target, false, outTensor);
        auto op = new AliasOp(getThis(), outTensor.get(), lazy);
        realizeOp(op, lazy);
        return outTensor;
    }

    TensorPtr Tensor::diffAlias(bool lazy, TensorPtr outTensor) {
        outTensor = initTensor(shape, false, outTensor);
        auto op = new DiffAliasOp(getThis(), outTensor.get(), lazy);
        realizeOp(op, lazy);
        return outTensor;
    }

    TensorPtr Tensor::copy(bool lazy, TensorPtr outTensor) {
        outTensor = initTensor(shape, true, outTensor);
        auto op = new CopyOp(getThis(), outTensor.get(), lazy);
        realizeOp(op, lazy);
        return outTensor;
    }

    TensorPtr Tensor::squeeze(int64_t dim, bool lazy, TensorPtr outTensor) {
        Shape outShape;

        if (dim != -1) {
            assert(Error::str_assert(isDimValid(dim), Error::Message::invalidDim(dim, shape)));
            outShape = shape;

            if (outShape[dim] == 1 && outShape.getNumDims() > 1) {
                outShape.remove(dim);
            }
        } else {
            outShape = shape;
            int i = 0;

            while (i < outShape.getNumDims()) {
                if (outShape[i] == 1 && outShape.getNumDims() > 1) {
                    outShape.remove(i);
                } else {
                    i++;
                }
            }
        }

        return alias(outShape, lazy, std::move(outTensor));
    }

    TensorPtr Tensor::unsqueeze(int64_t dim, bool lazy, TensorPtr outTensor) {
        assert(Error::str_assert(isDimValid(dim), Error::Message::invalidDim(dim, shape)));
        Shape outShape = shape;

        if (dim == -1) {
            outShape.view.push_back(1);
            outShape.strides.push_back(1);
        } else {
            outShape.view.insert(outShape.view.begin() + dim, 1);
            size_t stride = outShape.view[dim] * outShape.strides[dim];
            outShape.strides.insert(outShape.strides.begin() + dim, stride);
        }

        return alias(outShape, lazy, std::move(outTensor));
    }

    TensorPtr Tensor::at(const std::vector<Range> &ranges, bool lazy, TensorPtr outTensor) {
        assert(Error::str_assert(shape.getNumDims() >= ranges.size(), Error::Message::indexMultidimsOnly));
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

        return index(newRanges, lazy, std::move(outTensor));
    }

    TensorPtr Tensor::arange(const Shape &shape, real start, real step, bool lazy, TensorPtr outTensor) {
        outTensor = initTensor(shape, true, outTensor);
        auto op = new ArangeOp(outTensor.get(), start, step, lazy);
        realizeOp(op, lazy);
        return outTensor;
    }

    TensorPtr Tensor::randint(const Shape &shape, int64_t min, int64_t max, bool lazy, TensorPtr outTensor) {
        outTensor = initTensor(shape, true, outTensor);
        auto op = new RandintOp(outTensor.get(), min, max, lazy);
        realizeOp(op, lazy);
        return outTensor;
    }

    TensorPtr Tensor::randn(const Shape &shape, bool lazy, TensorPtr outTensor) {
        outTensor = initTensor(shape, true, outTensor);
        auto op = new RandnOp(outTensor.get(), lazy);
        realizeOp(op, lazy);
        return outTensor;
    }

    TensorPtr Tensor::fromConst(const Shape &shape, real c, bool lazy, TensorPtr outTensor) {
        outTensor = initTensor(shape, true, outTensor);
        auto op = new ConstOp(outTensor.get(), c, lazy);
        realizeOp(op, lazy);
        return outTensor;
    }

    TensorPtr Tensor::fromArr(const Shape &shape, const real *data, bool lazy, TensorPtr outTensor) {
        outTensor = initTensor(shape, true, outTensor);
        auto op = new FromArrOp(outTensor.get(), data, lazy);
        realizeOp(op, lazy);
        return outTensor;
    }

    TensorPtr Tensor::operator[](size_t idx) {
        auto outTensor = at(idx, true, nullptr);
        return outTensor;
    }

    TensorPtr Tensor::add(Tensor &rhs, bool lazy, TensorPtr outTensor) {
        assert(Error::str_assert(rhs.isBroadcastableTo(shape),
            Error::Message::notBroadcastable(rhs.shape, shape)));
        auto broadcastedRhs = rhs.broadcastTo(shape, lazy, nullptr);
        outTensor = initTensor(shape, true, outTensor);
        auto op = new AddOp(getThis(), broadcastedRhs, outTensor.get(), lazy);
        realizeOp(op, lazy);
        return outTensor;
    }

    TensorPtr Tensor::sub(Tensor &rhs, bool lazy, TensorPtr outTensor) {
        assert(Error::str_assert(rhs.isBroadcastableTo(shape),
            Error::Message::notBroadcastable(rhs.shape, shape)));
        auto broadcastedRhs = rhs.broadcastTo(shape, lazy, nullptr);
        outTensor = initTensor(shape, true, outTensor);
        auto op = new SubOp(getThis(), broadcastedRhs, outTensor.get(), lazy);
        realizeOp(op, lazy);
        return outTensor;
    }

    TensorPtr Tensor::mul(Tensor &rhs, bool lazy, TensorPtr outTensor) {
        assert(Error::str_assert(rhs.isBroadcastableTo(shape),
            Error::Message::notBroadcastable(rhs.shape, shape)));
        auto broadcastedRhs = rhs.broadcastTo(shape, lazy, nullptr);
        outTensor = initTensor(shape, true, outTensor);
        auto op = new MulOp(getThis(), broadcastedRhs, outTensor.get(), lazy);
        realizeOp(op, lazy);
        return outTensor;
    }

    TensorPtr Tensor::div(Tensor &rhs, bool lazy, TensorPtr outTensor) {
        assert(Error::str_assert(rhs.isBroadcastableTo(shape),
            Error::Message::notBroadcastable(rhs.shape, shape)));
        auto broadcastedRhs = rhs.broadcastTo(shape, lazy, nullptr);
        outTensor = initTensor(shape, true, outTensor);
        auto op = new DivOp(getThis(), broadcastedRhs, outTensor.get(), lazy);
        realizeOp(op, lazy);
        return outTensor;
    }

    TensorPtr Tensor::pow(real c, bool lazy, TensorPtr outTensor) {
        outTensor = initTensor(shape, true, outTensor);
        auto op = new PowOp(getThis(), outTensor.get(), c, lazy);
        realizeOp(op, lazy);
        return outTensor;
    }

    TensorPtr Tensor::log(bool lazy, TensorPtr outTensor) {
        outTensor = initTensor(shape, true, outTensor);
        auto op = new LogOp(getThis(), outTensor.get(), lazy);
        realizeOp(op, lazy);
        return outTensor;
    }

    TensorPtr Tensor::sin(bool lazy, TensorPtr outTensor) {
        outTensor = initTensor(shape, true, outTensor);
        auto op = new SinOp(getThis(), outTensor.get(), lazy);
        realizeOp(op, lazy);
        return outTensor;
    }

    TensorPtr Tensor::cos(bool lazy, TensorPtr outTensor) {
        outTensor = initTensor(shape, true, outTensor);
        auto op = new CosOp(getThis(), outTensor.get(), lazy);
        realizeOp(op, lazy);
        return outTensor;
    }

    TensorPtr Tensor::exp(bool lazy, TensorPtr outTensor) {
        outTensor = initTensor(shape, true, outTensor);
        auto op = new ExpOp(getThis(), outTensor.get(), lazy);
        realizeOp(op, lazy);
        return outTensor;
    }

    TensorPtr Tensor::recip(real c, bool lazy, TensorPtr outTensor) {
        outTensor = initTensor(shape, true, outTensor);
        auto op = new RecipOp(getThis(), outTensor.get(), c, lazy);
        realizeOp(op, lazy);
        return outTensor;
    }

    TensorPtr Tensor::sq(bool lazy, TensorPtr outTensor) {
        outTensor = initTensor(shape, true, outTensor);
        auto op = new SqOp(getThis(), outTensor.get(), lazy);
        realizeOp(op, lazy);
        return outTensor;
    }

    TensorPtr Tensor::sqrt(bool lazy, TensorPtr outTensor) {
        outTensor = initTensor(shape, true, outTensor);
        auto op = new SqrtOp(getThis(), outTensor.get(), lazy);
        realizeOp(op, lazy);
        return outTensor;
    }

    TensorPtr Tensor::neg(bool lazy, TensorPtr outTensor) {
        outTensor = initTensor(shape, true, outTensor);
        auto op = new NegOp(getThis(), outTensor.get(), lazy);
        realizeOp(op, lazy);
        return outTensor;
    }

    TensorPtr Tensor::eq(Tensor &rhs, bool lazy, TensorPtr outTensor) {
        assert(Error::str_assert(rhs.isBroadcastableTo(shape),
            Error::Message::notBroadcastable(rhs.shape, shape)));
        auto broadcastedRhs = rhs.broadcastTo(shape, lazy, nullptr);
        outTensor = initTensor(shape, true, outTensor);
        auto op = new EqOp(getThis(), broadcastedRhs, outTensor.get(), lazy);
        realizeOp(op, lazy);
        return outTensor;
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

    TensorPtr Tensor::neq(Tensor &rhs, bool lazy, TensorPtr outTensor) {
        assert(Error::str_assert(rhs.isBroadcastableTo(shape),
            Error::Message::notBroadcastable(rhs.shape, shape)));
        auto broadcastedRhs = rhs.broadcastTo(shape, lazy, nullptr);
        outTensor = initTensor(shape, true, outTensor);
        auto op = new NeqOp(getThis(), broadcastedRhs, outTensor.get(), lazy);
        realizeOp(op, lazy);
        return outTensor;
    }

    bool Tensor::operator!=(const Tensor &rhs) const {
        return !(*this == rhs);
    }

    TensorPtr Tensor::lt(Tensor &rhs, bool lazy, TensorPtr outTensor) {
        assert(Error::str_assert(rhs.isBroadcastableTo(shape),
            Error::Message::notBroadcastable(rhs.shape, shape)));
        auto broadcastedRhs = rhs.broadcastTo(shape, lazy, nullptr);
        outTensor = initTensor(shape, true, outTensor);
        auto op = new LessOp(getThis(), broadcastedRhs, outTensor.get(), lazy);
        realizeOp(op, lazy);
        return outTensor;
    }

    TensorPtr Tensor::gt(Tensor &rhs, bool lazy, TensorPtr outTensor) {
        assert(Error::str_assert(rhs.isBroadcastableTo(shape),
            Error::Message::notBroadcastable(rhs.shape, shape)));
        auto broadcastedRhs = rhs.broadcastTo(shape, lazy, nullptr);
        outTensor = initTensor(shape, true, outTensor);
        auto op = new GreaterOp(getThis(), broadcastedRhs, outTensor.get(), lazy);
        realizeOp(op, lazy);
        return outTensor;
    }

    TensorPtr Tensor::leq(Tensor &rhs, bool lazy, TensorPtr outTensor) {
        assert(Error::str_assert(rhs.isBroadcastableTo(shape),
            Error::Message::notBroadcastable(rhs.shape, shape)));
        auto broadcastedRhs = rhs.broadcastTo(shape, lazy, nullptr);
        outTensor = initTensor(shape, true, outTensor);
        auto op = new LeqOp(getThis(), broadcastedRhs, outTensor.get(), lazy);
        realizeOp(op, lazy);
        return outTensor;
    }

    TensorPtr Tensor::geq(Tensor &rhs, bool lazy, TensorPtr outTensor) {
        assert(Error::str_assert(rhs.isBroadcastableTo(shape),
            Error::Message::notBroadcastable(rhs.shape, shape)));
        auto broadcastedRhs = rhs.broadcastTo(shape, lazy, nullptr);
        outTensor = initTensor(shape, true, outTensor);
        auto op = new GeqOp(getThis(), broadcastedRhs, outTensor.get(), lazy);
        realizeOp(op, lazy);
        return outTensor;
    }

    TensorPtr Tensor::addAssign(Tensor &rhs, bool lazy) {
        assert(Error::str_assert(rhs.isBroadcastableTo(shape),
            Error::Message::notBroadcastable(rhs.shape, shape)));
        auto broadcastedRhs = rhs.broadcastTo(shape, lazy, nullptr);
        auto op = new AddAssignOp(broadcastedRhs, this, lazy);
        realizeOp(op, lazy);
        return getThis();
    }

    TensorPtr Tensor::subAssign(Tensor &rhs, bool lazy) {
        assert(Error::str_assert(rhs.isBroadcastableTo(shape),
            Error::Message::notBroadcastable(rhs.shape, shape)));
        auto broadcastedRhs = rhs.broadcastTo(shape, lazy, nullptr);
        auto op = new SubAssignOp(broadcastedRhs, this, lazy);
        realizeOp(op, lazy);
        return getThis();
    }

    TensorPtr Tensor::mulAssign(Tensor &rhs, bool lazy) {
        assert(Error::str_assert(rhs.isBroadcastableTo(shape),
            Error::Message::notBroadcastable(rhs.shape, shape)));
        auto broadcastedRhs = rhs.broadcastTo(shape, lazy, nullptr);
        auto op = new MulAssignOp(broadcastedRhs, this, lazy);
        realizeOp(op, lazy);
        return getThis();
    }

    TensorPtr Tensor::divAssign(Tensor &rhs, bool lazy) {
        assert(Error::str_assert(rhs.isBroadcastableTo(shape),
            Error::Message::notBroadcastable(rhs.shape, shape)));
        auto broadcastedRhs = rhs.broadcastTo(shape, lazy, nullptr);
        auto op = new DivAssignOp(broadcastedRhs, this, lazy);
        realizeOp(op, lazy);
        return getThis();
    }

    TensorPtr Tensor::relu(bool lazy, TensorPtr outTensor) {
        outTensor = initTensor(shape, true, outTensor);
        auto op = new ReluOp(getThis(), outTensor.get(), lazy);
        realizeOp(op, lazy);
        return outTensor;
    }

    TensorPtr Tensor::sigmoid(bool lazy, TensorPtr outTensor) {
        outTensor = initTensor(shape, true, outTensor);
        auto op = new SigmoidOp(getThis(), outTensor.get(), lazy);
        realizeOp(op, lazy);
        return outTensor;
    }

    TensorPtr Tensor::softmax(int64_t dim, bool lazy, TensorPtr outTensor) {
        assert(Error::str_assert(isDimValid(dim), Error::Message::invalidDim(dim, shape)));

        if (dim == -1) {
            auto maxTensor = max(-1, lazy, nullptr);
            auto subTensor = sub(maxTensor, lazy, nullptr);
            auto expTensor = subTensor->exp(lazy, nullptr);
            auto sumTensor = expTensor->sum(-1, lazy, nullptr);
            outTensor = expTensor->div(sumTensor, lazy, outTensor);
        } else {
            std::vector<size_t> shapePerm(shape.getNumDims());
            std::iota(shapePerm.begin(), shapePerm.end(), 0);
            shapePerm.erase(shapePerm.begin() + dim);
            shapePerm.push_back(dim);
            // Permutes operand
            Shape permShape = shape.perm(shapePerm);
            auto permTensor = perm(permShape, lazy, nullptr);
            // Compute softmax
            auto maxTensor = permTensor->max(shape.getNumDims() - 1, lazy, nullptr)->unsqueeze(-1, lazy, nullptr);
            auto subTensor = permTensor->sub(maxTensor, lazy, nullptr);
            auto expTensor = subTensor->exp(lazy, nullptr);
            auto sumTensor = expTensor->sum(shape.getNumDims() - 1, lazy, nullptr)->unsqueeze(-1, lazy, nullptr);
            auto softmaxTensor = expTensor->div(sumTensor, lazy, nullptr);
            // Permute softmax
            std::iota(shapePerm.begin(), shapePerm.end(), 0);
            size_t lastDim = shapePerm[shapePerm.size() - 1];
            shapePerm.erase(shapePerm.end() - 1);
            shapePerm.insert(shapePerm.begin() + dim, lastDim);
            outTensor = softmaxTensor->perm(shapePerm, lazy, outTensor);
        }

        return outTensor;
    }

    TensorPtr Tensor::matmul(Tensor &rhs, bool lazy, TensorPtr outTensor) {
        const auto message = Error::Message::shapesMismatched("matmul", shape, rhs.shape);
        assert(Error::str_assert(shape.getNumDims() == rhs.shape.getNumDims(), message));
        assert(Error::str_assert(shape.getNumDims() >= 2, Error::Message::matmulOnLessThan2d));
        assert(Error::str_assert(std::equal(shape.begin(), shape.end() - 2,
            rhs.shape.begin(), rhs.shape.end() - 2), message));
        size_t numDims = shape.getNumDims();
        assert(Error::str_assert(shape[numDims - 1] == rhs.shape[numDims - 2], message));
        Shape outShape = shape;
        // Shape of ...x H1 x W1 matmul ...x W1 x H2 == ...x H1 x H2
        outShape[numDims - 1] = rhs.shape[numDims - 1];
        // Permutes rhs's last two dimensions
        auto tranposedRhs = rhs.T(numDims - 2, lazy);
        // Do matrix multiplication on the last 2 dimensions
        outTensor = initTensor(outShape, true, outTensor);
        auto op = new MatmulOp(getThis(), tranposedRhs, outTensor.get(), lazy);
        realizeOp(op, lazy);
        return outTensor;
    }

    TensorPtr Tensor::reshape(const Shape &target, bool lazy, TensorPtr outTensor) {
        assert(Error::str_assert(target.getSize() == shape.getSize(),
            Error::Message::shapesMismatched("matmul", shape, target)));

        if (isContiguous()) {
            outTensor = initTensor(target, false, outTensor);
            outTensor->shape.offset = shape.offset;
            auto op = new AliasOp(getThis(), outTensor.get(), lazy);
            realizeOp(op, lazy);
        } else {
            outTensor = initTensor(target, true, outTensor);
            auto op = new CopyOp(getThis(), outTensor.get(), lazy);
            realizeOp(op, lazy);
        }

        return outTensor;
    }

    TensorPtr Tensor::sum(int64_t dim, bool lazy, TensorPtr outTensor) {
        assert(Error::str_assert(isDimValid(dim), Error::Message::invalidDim(dim, shape)));

        if (dim == -1) {
            outTensor = initTensor(Shape({1}), true, outTensor);
            auto op = new SumOp(getThis(), outTensor.get(), dim, lazy);
            realizeOp(op, lazy);
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
            auto permTensor = perm(shapePerm, lazy, nullptr);
            outTensor = initTensor(outShape, true, outTensor);
            auto op = new SumOp(permTensor, outTensor.get(), dim, lazy);
            realizeOp(op, lazy);
        }

        return outTensor;
    }

    TensorPtr Tensor::max(int64_t dim, bool lazy, TensorPtr outTensor) {
        assert(Error::str_assert(isDimValid(dim), Error::Message::invalidDim(dim, shape)));

        if (dim == -1) {
            outTensor = initTensor(Shape({1}), true, outTensor);
            auto op = new MaxOp(getThis(), outTensor.get(), dim, lazy);
            realizeOp(op, lazy);
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
            auto permTensor = perm(shapePerm, lazy, nullptr);
            outTensor = initTensor(outShape, true, outTensor);
            auto op = new MaxOp(permTensor, outTensor.get(), dim, lazy);
            realizeOp(op, lazy);
        }

        return outTensor;
    }

    TensorPtr Tensor::min(int64_t dim, bool lazy, TensorPtr outTensor) {
        assert(Error::str_assert(isDimValid(dim), Error::Message::invalidDim(dim, shape)));

        if (dim == -1) {
            outTensor = initTensor(Shape({1}), true, outTensor);
            auto op = new MinOp(getThis(), outTensor.get(), dim, lazy);
            realizeOp(op, lazy);
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
            auto permTensor = perm(shapePerm, lazy, nullptr);
            outTensor = initTensor(outShape, true, outTensor);
            auto op = new MinOp(permTensor, outTensor.get(), dim, lazy);
            realizeOp(op, lazy);
        }

        return outTensor;
    }

    TensorPtr Tensor::perm(const std::vector<size_t> &shapePerm, bool lazy, TensorPtr outTensor) {
        assert(Error::str_assert(shapePerm.size() == shape.getNumDims(), Error::Message::invalidShapePerm));
        std::vector<bool> flags(shape.getNumDims());

        for (size_t i: shapePerm) {
            assert(Error::str_assert(i < shape.getNumDims(), Error::Message::invalidShapePerm));
            flags[i] = true;
        }

        for (bool i: flags) {
            assert(Error::str_assert(i, Error::Message::invalidShapePerm));
        }

        Shape permShape = shape.perm(shapePerm);
        outTensor = initTensor(permShape, false, outTensor);
        auto op = new PermOp(getThis(), outTensor.get(), lazy);
        realizeOp(op, lazy);
        return outTensor;
    }

    TensorPtr Tensor::perm(const Shape &target, bool lazy, TensorPtr outTensor) {
        outTensor = initTensor(target, false, outTensor);
        auto op = new PermOp(getThis(), outTensor.get(), lazy);
        realizeOp(op, lazy);
        return outTensor;
    }

    TensorPtr Tensor::T(size_t startDim, bool lazy, TensorPtr outTensor) {
        assert(Error::str_assert(startDim < shape.getNumDims(), Error::Message::invalidDim(startDim, shape)));
        std::vector<size_t> shapePerm(shape.getNumDims());
        std::iota(shapePerm.begin(), shapePerm.end(), 0);
        std::reverse(shapePerm.begin() + startDim, shapePerm.end());
        return perm(shapePerm, lazy, std::move(outTensor));
    }

    bool Tensor::isEmpty() const {
        IterPtr iter = initConstIter(this);
        iter->start();
        return !iter->hasNext();
    }

    void Tensor::forward() {
        if (graph == nullptr) {
            graph = new TensorGraph(this);
        }

        graph->forward();
    }

    void Tensor::backward() {
        assert(Error::str_assert(graph != nullptr, Error::Message::tensorGraphUninitialized));
        graph->backward();
    }
}
