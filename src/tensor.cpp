#include <iostream>

#include "tensor.h"
#include "ops.h"
#include "tensor_iter.h"
#include "cgraph.h"

namespace Toygrad::Tensor {
    size_t Tensor::idCounter = 0;

    Tensor::Tensor() {
        id = idCounter++;
    }

    Tensor::Tensor(const Shape &shape) : Tensor() {
        this->shape = shape;
    }

    Tensor::~Tensor() {
        std::cout << "Destroyed tensor " << id << "..." << std::endl;
        for (const auto &op: ops) delete op;
        ops.clear();
    }

    std::ostream &operator<<(std::ostream &stream, Tensor &tensor) {
        IterPtr iter = initIter(&tensor);
        std::vector<size_t> subsize(tensor.shape.getNumDims());
        size_t tmp = 1;

        for (int i = subsize.size() - 1; i >= 0; i--) {
            tmp *= tensor.shape.view[i];
            subsize[i] = tmp;
        }

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

            for (int i = subsize.size() - 1; i >= 0; i--) {
                if (iter->count() % subsize[i] == 0) {
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

    TensorPtr Tensor::index(const std::vector<size_t> &idx) {
        assert(str_assert(shape.getNumDims() > idx.size(), AssertMessage::indexMultidimsOnly));

        // Index must stay within bounds
        for (size_t i = 0; i < idx.size(); i++) {
            assert(str_assert(idx[i] < shape.view[i], AssertMessage::indexOutOfBounds));
        }

        auto ranges = std::vector<Range>();

        for (size_t i: idx) {
            ranges.push_back({i, i + 1, 1});
        }

        for (size_t i = idx.size(); i < shape.getNumDims(); i++) {
            ranges.push_back({0, shape[i], 1});
        }

        return index(ranges);
    }

    TensorPtr Tensor::index(const std::vector<Range> &ranges) {
        Shape outShape;
        outShape.offset = shape.offset;
        outShape.ranges = ranges;

        for (size_t i = 0; i < ranges.size(); i++) {
            outShape.offset += ranges[i].beg * shape.strides[i];
        }

        for (size_t i = 0; i < ranges.size(); i++) {
            size_t dim = ceil(static_cast<real>(ranges[i].end - ranges[i].beg) / ranges[i].step);
            outShape.view.push_back(dim);
            outShape.strides.push_back(shape.strides[i] * ranges[i].step);
        }

        auto outTensor = std::make_shared<Tensor>(outShape);
        outTensor->ops.push_back(new AliasOp(shared_from_this(), outTensor.get()));
        outTensor->ops.back()->forward();
        return outTensor;
    }

    bool Tensor::isContiguous() const {
        return std::ranges::all_of(shape.ranges.begin(), shape.ranges.end(),
                                   [](const Range &range) { return range.step > 1; });
    }

    bool Tensor::isBroadcastableTo(const Shape &target) const {
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
        assert(str_assert(isBroadcastableTo(target), AssertMessage::notBroadcastable));
        Shape outShape;
        outShape.offset = shape.offset;
        outShape.view = shape.view;
        size_t dimsToAdd = target.getNumDims() - outShape.getNumDims();

        for (size_t i = 0; i < dimsToAdd; i++) {
            outShape.view.insert(outShape.view.begin(), 1);
        }

        outShape.defStrides();

        for (int i = target.getNumDims() - 1; i >= 0; i--) {
            if (outShape.view[i] < target.view[i]) {
                // outShape.view[i] == 1
                outShape.view[i] = target.view[i];
                outShape.strides[i] = 0;
            }
        }

        outShape.defRanges();
        auto outTensor = std::make_shared<Tensor>(outShape);
        outTensor->ops.push_back(new AliasOp(shared_from_this(), outTensor.get()));
        outTensor->ops.back()->forward();
        return outTensor;
    }

    TensorPtr Tensor::alias() {
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->ops.push_back(new AliasOp(shared_from_this(), outTensor.get()));
        outTensor->ops.back()->forward();
        return outTensor;
    }

    TensorPtr Tensor::copy() {
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->ops.push_back(new CopyOp(shared_from_this(), outTensor.get()));
        outTensor->ops.back()->forward();
        return outTensor;
    }

    bool Tensor::isSqueezable(int dim) const {
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

    TensorPtr Tensor::squeeze(int dim) {
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

        auto outTensor = std::make_shared<Tensor>(outShape);
        outTensor->ops.push_back(new AliasOp(shared_from_this(), outTensor.get()));
        outTensor->ops.back()->forward();
        return outTensor;
    }

    TensorPtr Tensor::unsqueeze(int dim) {
        assert(str_assert(isDimValid(dim), AssertMessage::invalidDim));
        Shape outShape = shape;

        if (dim == -1) {
            outShape.view.push_back(1);
            outShape.strides.push_back(1);
            outShape.ranges.push_back(Range(0, 1, 1));
        }
    }

    TensorPtr Tensor::at(const std::vector<size_t> &idx) {
        return index(idx);
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
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->ops.push_back(new ArangeOp(outTensor.get(), start, step));
        outTensor->ops.back()->forward();
        return outTensor;
    }

    TensorPtr Tensor::randint(const Shape &shape, int min, int max) {
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->ops.push_back(new RandintOp(outTensor.get(), min, max));
        outTensor->ops.back()->forward();
        return outTensor;
    }

    TensorPtr Tensor::randn(const Shape &shape) {
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->ops.push_back(new RandnOp(outTensor.get()));
        outTensor->ops.back()->forward();
        return outTensor;
    }

    TensorPtr Tensor::fromConst(const Shape &shape, real c) {
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->ops.push_back(new ConstOp(outTensor.get(), c));
        outTensor->ops.back()->forward();
        return outTensor;
    }

    TensorPtr Tensor::fromArr(const Shape &shape, const real *data) {
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->ops.push_back(new FromArrOp(outTensor.get(), data));
        outTensor->ops.back()->forward();
        return outTensor;
    }

    TensorPtr Tensor::operator[](size_t idx) {
        auto outTensor = index(std::vector(1, idx));
        return outTensor;
    }

    TensorPtr Tensor::add(const TensorPtr &rhs) {
        assert(str_assert(shape == rhs->shape, AssertMessage::shapesMismatched));
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->ops.push_back(new AddOp(shared_from_this(), rhs, outTensor.get()));
        outTensor->ops.back()->forward();
        return outTensor;
    }

    TensorPtr Tensor::add(real c) {
        return add(fromConst(shape, c));
    }

    TensorPtr Tensor::sub(const TensorPtr &rhs) {
        assert(str_assert(shape == rhs->shape, AssertMessage::shapesMismatched));
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->ops.push_back(new SubOp(shared_from_this(), rhs, outTensor.get()));
        outTensor->ops.back()->forward();
        return outTensor;
    }

    TensorPtr Tensor::sub(real c) {
        return sub(fromConst(shape, c));
    }

    TensorPtr Tensor::mul(const TensorPtr &rhs) {
        assert(str_assert(shape == rhs->shape, AssertMessage::shapesMismatched));
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->ops.push_back(new MulOp(shared_from_this(), rhs, outTensor.get()));
        outTensor->ops.back()->forward();
        return outTensor;
    }

    TensorPtr Tensor::mul(real c) {
        return mul(fromConst(shape, c));
    }

    TensorPtr Tensor::div(const TensorPtr &rhs) {
        assert(str_assert(shape == rhs->shape, AssertMessage::shapesMismatched));
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->ops.push_back(new DivOp(shared_from_this(), rhs, outTensor.get()));
        outTensor->ops.back()->forward();
        return outTensor;
    }

    TensorPtr Tensor::div(real c) {
        return div(fromConst(shape, c));
    }

    TensorPtr Tensor::pow(real c) {
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->ops.push_back(new PowOp(shared_from_this(), outTensor.get(), c));
        outTensor->ops.back()->forward();
        return outTensor;
    }

    TensorPtr Tensor::log() {
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->ops.push_back(new LogOp(shared_from_this(), outTensor.get()));
        outTensor->ops.back()->forward();
        return outTensor;
    }

    TensorPtr Tensor::sin() {
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->ops.push_back(new SinOp(shared_from_this(), outTensor.get()));
        outTensor->ops.back()->forward();
        return outTensor;
    }

    TensorPtr Tensor::cos() {
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->ops.push_back(new CosOp(shared_from_this(), outTensor.get()));
        outTensor->ops.back()->forward();
        return outTensor;
    }

    TensorPtr Tensor::exp() {
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->ops.push_back(new ExpOp(shared_from_this(), outTensor.get()));
        outTensor->ops.back()->forward();
        return outTensor;
    }

    TensorPtr Tensor::recip(real c) {
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->ops.push_back(new RecipOp(shared_from_this(), outTensor.get(), c));
        outTensor->ops.back()->forward();
        return outTensor;
    }

    TensorPtr Tensor::sq() {
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->ops.push_back(new SqOp(shared_from_this(), outTensor.get()));
        outTensor->ops.back()->forward();
        return outTensor;
    }

    TensorPtr Tensor::sqrt() {
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->ops.push_back(new SqrtOp(shared_from_this(), outTensor.get()));
        outTensor->ops.back()->forward();
        return outTensor;
    }

    TensorPtr Tensor::neg() {
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->ops.push_back(new NegOp(shared_from_this(), outTensor.get()));
        outTensor->ops.back()->forward();
        return outTensor;
    }

    TensorPtr Tensor::eq(const TensorPtr &rhs) {
        assert(str_assert(shape == rhs->shape, AssertMessage::shapesMismatched));
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->ops.push_back(new EqOp(shared_from_this(), rhs, outTensor.get()));
        outTensor->ops.back()->forward();
        return outTensor;
    }

    TensorPtr Tensor::eq(real c) {
        return eq(fromConst(shape, c));
    }

    bool Tensor::operator==(const Tensor &rhs) const {
        if (shape != rhs.shape) {
            return false;
        }

        SparseIter lhsIter(this);
        SparseIter rhsIter(&rhs);

        for (lhsIter.start(), rhsIter.start(); lhsIter.hasNext(); lhsIter.next(), rhsIter.next()) {
            if (lhsIter.curr() != rhsIter.curr()) {
                return false;
            }
        }

        return true;
    }

    TensorPtr Tensor::neq(const TensorPtr &rhs) {
        assert(str_assert(shape == rhs->shape, AssertMessage::shapesMismatched));
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->ops.push_back(new NeqOp(shared_from_this(), rhs, outTensor.get()));
        outTensor->ops.back()->forward();
        return outTensor;
    }

    TensorPtr Tensor::neq(real c) {
        return neq(fromConst(shape, c));
    }

    bool Tensor::operator!=(const Tensor &rhs) const {
        return !(*this == rhs);
    }

    TensorPtr Tensor::lt(const TensorPtr &rhs) {
        assert(str_assert(shape == rhs->shape, AssertMessage::shapesMismatched));
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->ops.push_back(new LessOp(shared_from_this(), rhs, outTensor.get()));
        outTensor->ops.back()->forward();
        return outTensor;
    }

    TensorPtr Tensor::lt(real c) {
        return lt(fromConst(shape, c));
    }

    TensorPtr Tensor::gt(const TensorPtr &rhs) {
        assert(str_assert(shape == rhs->shape, AssertMessage::shapesMismatched));
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->ops.push_back(new GreaterOp(shared_from_this(), rhs, outTensor.get()));
        outTensor->ops.back()->forward();
        return outTensor;
    }

    TensorPtr Tensor::gt(real c) {
        return gt(fromConst(shape, c));
    }

    TensorPtr Tensor::leq(const TensorPtr &rhs) {
        assert(str_assert(shape == rhs->shape, AssertMessage::shapesMismatched));
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->ops.push_back(new LeqOp(shared_from_this(), rhs, outTensor.get()));
        outTensor->ops.back()->forward();
        return outTensor;
    }

    TensorPtr Tensor::leq(real c) {
        return leq(fromConst(shape, c));
    }

    TensorPtr Tensor::geq(const TensorPtr &rhs) {
        assert(str_assert(shape == rhs->shape, AssertMessage::shapesMismatched));
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->ops.push_back(new GeqOp(shared_from_this(), rhs, outTensor.get()));
        outTensor->ops.back()->forward();
        return outTensor;
    }

    TensorPtr Tensor::geq(real c) {
        return geq(fromConst(shape, c));
    }

    TensorPtr Tensor::addAssign(const TensorPtr &rhs) {
        assert(str_assert(shape == rhs->shape, AssertMessage::shapesMismatched));
        ops.push_back(new AddAssignOp(rhs, this));
        ops.back()->forward();
        return shared_from_this();
    }

    TensorPtr Tensor::addAssign(real c) {
        return addAssign(fromConst(shape, c));
    }

    TensorPtr Tensor::subAssign(const TensorPtr &rhs) {
        assert(str_assert(shape == rhs->shape, AssertMessage::shapesMismatched));
        ops.push_back(new SubAssignOp(rhs, this));
        ops.back()->forward();
        return shared_from_this();
    }

    TensorPtr Tensor::subAssign(real c) {
        return subAssign(fromConst(shape, c));
    }

    TensorPtr Tensor::mulAssign(const TensorPtr &rhs) {
        assert(str_assert(shape == rhs->shape, AssertMessage::shapesMismatched));
        ops.push_back(new MulAssignOp(rhs, this));
        ops.back()->forward();
        return shared_from_this();
    }

    TensorPtr Tensor::mulAssign(real c) {
        return mulAssign(fromConst(shape, c));
    }

    TensorPtr Tensor::divAssign(const TensorPtr &rhs) {
        assert(str_assert(shape == rhs->shape, AssertMessage::shapesMismatched));
        ops.push_back(new DivAssignOp(rhs, this));
        ops.back()->forward();
        return shared_from_this();
    }

    TensorPtr Tensor::divAssign(real c) {
        return divAssign(fromConst(shape, c));
    }

    TensorPtr Tensor::relu() {
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->ops.push_back(new ReluOp(shared_from_this(), outTensor.get()));
        outTensor->ops.back()->forward();
        return outTensor;
    }

    TensorPtr Tensor::sigmoid() {
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->ops.push_back(new SigmoidOp(shared_from_this(), outTensor.get()));
        outTensor->ops.back()->forward();
        return outTensor;
    }

    TensorPtr Tensor::softmax(int dim) {
        assert(str_assert(isDimValid(dim), AssertMessage::invalidDim));
        TensorPtr outTensor;

        if (dim == -1) {
            Shape softmaxShape = shape;
            softmaxShape.offset = 0;
            outTensor = std::make_shared<Tensor>(softmaxShape);
            outTensor->ops.push_back(new SoftmaxOp(shared_from_this(), outTensor.get(), dim));
            outTensor->ops.back()->forward();
        } else {
            std::vector<size_t> shapePerm;

            for (size_t i = 0; i < shape.getNumDims(); i++) {
                if (i != dim) {
                    shapePerm.push_back(i);
                }
            }

            shapePerm.push_back(dim);
            // Permutate operand
            auto permTensor = perm(shapePerm);
            // Compute softmax
            Shape softmaxShape = shape.perm(shapePerm);
            softmaxShape.offset = 0;
            auto softmaxTensor = std::make_shared<Tensor>(softmaxShape);
            softmaxTensor->ops.push_back(new SoftmaxOp(permTensor, softmaxTensor.get(), dim));
            softmaxTensor->ops.back()->forward();
            // Permutate softmax
            Shape softmaxPermShape = shape;
            softmaxPermShape.offset = 0;
            outTensor = softmaxTensor->perm(softmaxPermShape);
        }

        return outTensor;
    }

    TensorPtr Tensor::reshape(const Shape &target) {
        assert(str_assert(target.getSize() == this->shape.getSize(), AssertMessage::shapesMismatched));
        TensorPtr outTensor;

        if (isContiguous()) {
            outTensor = std::make_shared<Tensor>(target);
            outTensor->shape.offset = this->shape.offset;
            outTensor->ops.push_back(new AliasOp(shared_from_this(), outTensor.get()));
            outTensor->ops.back()->forward();
        } else {
            outTensor = std::make_shared<Tensor>(target);
            outTensor->ops.push_back(new CopyOp(shared_from_this(), outTensor.get()));
            outTensor->ops.back()->forward();
        }

        return outTensor;
    }

    TensorPtr Tensor::sum(int dim) {
        assert(str_assert(isDimValid(dim), AssertMessage::invalidDim));
        TensorPtr outTensor;

        if (dim == -1) {
            outTensor = std::make_shared<Tensor>(Shape({1}));
            outTensor->ops.push_back(new SumOp(shared_from_this(), outTensor.get(), dim));
            outTensor->ops.back()->forward();
        } else {
            Shape sumShape;
            sumShape.offset = 0;
            std::vector<size_t> shapePerm;

            for (size_t i = 0; i < shape.getNumDims(); i++) {
                if (i != dim) {
                    sumShape.view.push_back(shape[i]);
                    shapePerm.push_back(i);
                }
            }

            sumShape.defRanges();
            sumShape.defStrides();
            shapePerm.push_back(dim);
            auto permTensor = perm(shapePerm);
            outTensor = std::make_shared<Tensor>(sumShape);
            outTensor->ops.push_back(new SumOp(permTensor, outTensor.get(), dim));
            outTensor->ops.back()->forward();
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
        auto outTensor = std::make_shared<Tensor>(permShape);
        outTensor->ops.push_back(new PermOp(shared_from_this(), outTensor.get()));
        outTensor->ops.back()->forward();
        return outTensor;
    }

    TensorPtr Tensor::perm(const Shape &target) {
        // TODO: check
        auto outTensor = std::make_shared<Tensor>(target);
        outTensor->ops.push_back(new PermOp(shared_from_this(), outTensor.get()));
        outTensor->ops.back()->forward();
        return outTensor;
    }

    TensorPtr Tensor::T() {
        std::vector<size_t> shapePerm(shape.getNumDims());

        for (size_t i = 0; i < shape.getNumDims(); i++) {
            shapePerm[i] = shape.getNumDims() - 1 - i;
        }

        return perm(shapePerm);
    }

    bool Tensor::isEmpty() {
        IterPtr iter = initIter(this);
        iter->start();
        return !iter->hasNext();
    }

    void Tensor::backward() {
        assert(str_assert(shape.getSize() == 1, AssertMessage::gradOnScalarOnly));
        static CGraph graph;
        graph.backprop(shared_from_this());
    }
}
