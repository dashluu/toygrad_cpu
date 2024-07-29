#include <iostream>

#include "tensor.h"
#include "ops.h"
#include "tensor_accessor.h"
#include "tensor_iter.h"
#include "cgraph.h"

namespace Toygrad::Tensor {
    size_t Tensor::idCounter = 0;

    Tensor::Tensor() {
        id = idCounter++;
    }

    Tensor::Tensor(const Shape &shape) : Tensor() {
        this->shape = shape;
        vec = std::make_shared<Vec>(shape.getSize());
    }

    Tensor::Tensor(const Shape &shape, const std::shared_ptr<Vec> &vec): Tensor() {
        this->shape = shape;
        this->vec = vec;
    }

    Tensor::Tensor(const Tensor &tensor): Tensor() {
        shape = tensor.shape;
        vec = tensor.vec;
    }

    Tensor::~Tensor() {
        std::cout << "Destroyed tensor " << id << "..." << std::endl;
        delete op;
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

    TensorPtr Tensor::atHelper(const std::vector<size_t> &idx) {
        assert(str_assert(shape.getNumDims() > idx.size(), AssertMessage::indexMultidimsOnly));

        // Index must stay within bounds
        for (size_t i = 0; i < idx.size(); i++) {
            assert(str_assert(idx[i] < shape.view[i], AssertMessage::indexOutOfBounds));
        }

        TensorAccessor indexer(this);
        return indexer.at(idx);
    }

    bool Tensor::isContiguous() const {
        return std::ranges::all_of(shape.ranges.begin(), shape.ranges.end(),
                                   [](const Range &range) { return range.step > 1; });
    }

    TensorPtr Tensor::at(const std::vector<size_t> &idx) {
        return atHelper(idx);
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

        TensorAccessor indexer(this);
        return indexer.at(newRanges);
    }

    TensorPtr Tensor::arange(const Shape &shape, real start, real step) {
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->op = new ArangeOp(outTensor.get(), start, step);
        outTensor->op->forward();
        return outTensor;
    }

    TensorPtr Tensor::randint(const Shape &shape, int min, int max) {
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->op = new RandintOp(outTensor.get(), min, max);
        outTensor->op->forward();
        return outTensor;
    }

    TensorPtr Tensor::randn(const Shape &shape) {
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->op = new RandnOp(outTensor.get());
        outTensor->op->forward();
        return outTensor;
    }

    TensorPtr Tensor::fromConst(const Shape &shape, real c) {
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->op = new ConstOp(outTensor.get(), c);
        outTensor->op->forward();
        return outTensor;
    }

    TensorPtr Tensor::fromArr(const Shape &shape, const real *data) {
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->op = new FromArrOp(outTensor.get(), data);
        outTensor->op->forward();
        return outTensor;
    }

    TensorPtr Tensor::operator[](size_t idx) {
        auto outTensor = atHelper(std::vector(1, idx));
        return outTensor;
    }

    TensorPtr Tensor::add(const TensorPtr &rhs) {
        assert(str_assert(shape == rhs->shape, AssertMessage::shapesMismatched));
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->op = new AddOp(shared_from_this(), rhs, outTensor.get());
        outTensor->op->forward();
        return outTensor;
    }

    TensorPtr Tensor::add(real c) {
        return add(fromConst(shape, c));
    }

    TensorPtr Tensor::sub(const TensorPtr &rhs) {
        assert(str_assert(shape == rhs->shape, AssertMessage::shapesMismatched));
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->op = new SubOp(shared_from_this(), rhs, outTensor.get());
        outTensor->op->forward();
        return outTensor;
    }

    TensorPtr Tensor::sub(real c) {
        return sub(fromConst(shape, c));
    }

    TensorPtr Tensor::mul(const TensorPtr &rhs) {
        assert(str_assert(shape == rhs->shape, AssertMessage::shapesMismatched));
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->op = new MulOp(shared_from_this(), rhs, outTensor.get());
        outTensor->op->forward();
        return outTensor;
    }

    TensorPtr Tensor::mul(real c) {
        return mul(fromConst(shape, c));
    }

    TensorPtr Tensor::div(const TensorPtr &rhs) {
        assert(str_assert(shape == rhs->shape, AssertMessage::shapesMismatched));
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->op = new DivOp(shared_from_this(), rhs, outTensor.get());
        outTensor->op->forward();
        return outTensor;
    }

    TensorPtr Tensor::div(real c) {
        return div(fromConst(shape, c));
    }

    TensorPtr Tensor::exp() {
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->op = new ExpOp(shared_from_this(), outTensor.get());
        outTensor->op->forward();
        return outTensor;
    }

    TensorPtr Tensor::recip(real c) {
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->op = new RecipOp(shared_from_this(), outTensor.get(), c);
        outTensor->op->forward();
        return outTensor;
    }

    TensorPtr Tensor::sq() {
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->op = new SqOp(shared_from_this(), outTensor.get());
        outTensor->op->forward();
        return outTensor;
    }

    TensorPtr Tensor::neg() {
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->op = new NegOp(shared_from_this(), outTensor.get());
        outTensor->op->forward();
        return outTensor;
    }

    TensorPtr Tensor::eq(const TensorPtr &rhs) {
        assert(str_assert(shape == rhs->shape, AssertMessage::shapesMismatched));
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->op = new EqOp(shared_from_this(), rhs, outTensor.get());
        outTensor->op->forward();
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
        outTensor->op = new NeqOp(shared_from_this(), rhs, outTensor.get());
        outTensor->op->forward();
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
        outTensor->op = new LessOp(shared_from_this(), rhs, outTensor.get());
        outTensor->op->forward();
        return outTensor;
    }

    TensorPtr Tensor::lt(real c) {
        return lt(fromConst(shape, c));
    }

    TensorPtr Tensor::gt(const TensorPtr &rhs) {
        assert(str_assert(shape == rhs->shape, AssertMessage::shapesMismatched));
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->op = new GreaterOp(shared_from_this(), rhs, outTensor.get());
        outTensor->op->forward();
        return outTensor;
    }

    TensorPtr Tensor::gt(real c) {
        return gt(fromConst(shape, c));
    }

    TensorPtr Tensor::leq(const TensorPtr &rhs) {
        assert(str_assert(shape == rhs->shape, AssertMessage::shapesMismatched));
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->op = new LeqOp(shared_from_this(), rhs, outTensor.get());
        outTensor->op->forward();
        return outTensor;
    }

    TensorPtr Tensor::leq(real c) {
        return leq(fromConst(shape, c));
    }

    TensorPtr Tensor::geq(const TensorPtr &rhs) {
        assert(str_assert(shape == rhs->shape, AssertMessage::shapesMismatched));
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->op = new GeqOp(shared_from_this(), rhs, outTensor.get());
        outTensor->op->forward();
        return outTensor;
    }

    TensorPtr Tensor::geq(real c) {
        return geq(fromConst(shape, c));
    }

    Tensor &Tensor::operator=(const Tensor &rhs) {
        if (this == &rhs) {
            return *this;
        }

        assert(str_assert(shape == rhs.shape, AssertMessage::shapesMismatched));
        vec = rhs.vec;
        // Keep the edges and op the same
        return *this;
    }

    Tensor &Tensor::operator=(real c) {
        auto constTensor = fromConst(shape, c);
        return *this = *constTensor;
    }

    TensorPtr Tensor::addAssign(const TensorPtr &rhs) {
        assert(str_assert(shape == rhs->shape, AssertMessage::shapesMismatched));
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->op = new AddAssignOp(shared_from_this(), rhs, outTensor.get());
        outTensor->op->forward();
        return outTensor;
    }

    TensorPtr Tensor::addAssign(real c) {
        return addAssign(fromConst(shape, c));
    }

    TensorPtr Tensor::subAssign(const TensorPtr &rhs) {
        assert(str_assert(shape == rhs->shape, AssertMessage::shapesMismatched));
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->op = new SubAssignOp(shared_from_this(), rhs, outTensor.get());
        outTensor->op->forward();
        return outTensor;
    }

    TensorPtr Tensor::subAssign(real c) {
        return subAssign(fromConst(shape, c));
    }

    TensorPtr Tensor::mulAssign(const TensorPtr &rhs) {
        assert(str_assert(shape == rhs->shape, AssertMessage::shapesMismatched));
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->op = new MulAssignOp(shared_from_this(), rhs, outTensor.get());
        outTensor->op->forward();
        return outTensor;
    }

    TensorPtr Tensor::mulAssign(real c) {
        return mulAssign(fromConst(shape, c));
    }

    TensorPtr Tensor::divAssign(const TensorPtr &rhs) {
        assert(str_assert(shape == rhs->shape, AssertMessage::shapesMismatched));
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->op = new DivAssignOp(shared_from_this(), rhs, outTensor.get());
        outTensor->op->forward();
        return outTensor;
    }

    TensorPtr Tensor::divAssign(real c) {
        return divAssign(fromConst(shape, c));
    }

    TensorPtr Tensor::relu() {
        auto outTensor = std::make_shared<Tensor>(shape);
        outTensor->op = new ReluOp(shared_from_this(), outTensor.get());
        outTensor->op->forward();
        return outTensor;
    }

    TensorPtr Tensor::reshape(const Shape &shape) {
        assert(str_assert(shape.getSize() == this->shape.getSize(), AssertMessage::shapesMismatched));
        TensorPtr outTensor;

        if (isContiguous()) {
            outTensor = std::make_shared<Tensor>(shape, vec);
            outTensor->shape.offset = this->shape.offset;
        } else {
            outTensor = std::make_shared<Tensor>(shape);
            outTensor->op = new CopyOp(shared_from_this(), outTensor.get());
            outTensor->op->forward();
        }

        return outTensor;
    }

    TensorPtr Tensor::sum(int dim) {
        assert(str_assert(dim >= -1 && dim < static_cast<int>(shape.getNumDims()), AssertMessage::invalidDim));
        TensorPtr outTensor;

        if (dim == -1) {
            outTensor = std::make_shared<Tensor>(Shape({1}));
            outTensor->op = new SumOp(shared_from_this(), nullptr, outTensor.get(), dim);
        } else {
            Shape sumShape;
            sumShape.offset = 0;
            std::vector<size_t> opPerm;

            for (size_t i = 0; i < shape.getNumDims(); i++) {
                if (i != dim) {
                    sumShape.view.push_back(shape[i]);
                    opPerm.push_back(i);
                }
            }

            sumShape.defRanges();
            sumShape.defStrides();
            opPerm.push_back(dim);
            outTensor = std::make_shared<Tensor>(sumShape);
            auto opTensor = perm(opPerm);
            outTensor->op = new SumOp(shared_from_this(), opTensor, outTensor.get(), dim);
        }

        outTensor->op->forward();
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

        Shape permShape(shape.offset, shape.view, shape.strides, shapePerm);
        return std::make_shared<Tensor>(permShape, vec);
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
