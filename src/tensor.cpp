#include <iostream>

#include "tensor.h"
#include "ops.h"
#include "tensor_indexer.h"
#include "tensor_iter.h"
#include "graph.h"

namespace Toygrad::Tensor {
    size_t Tensor::idCounter = 0;

    Tensor::Tensor() {
        id = idCounter++;
    }

    Tensor::Tensor(const Shape &shape) : Tensor() {
        this->shape = shape;
        vec = new Vec(shape.getSize());
        vec->refCount++;
    }

    Tensor::Tensor(const Shape &shape, Vec *vec): Tensor() {
        this->shape = shape;
        this->vec = vec;
        vec->refCount++;
    }

    Tensor::Tensor(const Tensor &tensor): Tensor() {
        shape = tensor.shape;
        vec = tensor.vec;
        vec->refCount++;
    }

    Tensor::~Tensor() {
        if (vec->refCount > 1) {
            vec->refCount--;
        } else {
            delete vec;
        }
    }

    Tensor &Tensor::getGrad() const {
        return *op->grad;
    }

    std::ostream &operator<<(std::ostream &stream, Tensor &tensor) {
        stream << "[";
        SparseIter iter(&tensor);

        for (iter.start(); iter.hasNext(); iter.next()) {
            stream << iter.curr() << " ";
        }

        stream << "]";
        return stream;
    }

    Tensor *Tensor::atHelper(const std::vector<size_t> &idx) {
        assert(str_assert(shape.getNumDims() > idx.size(), AssertMessage::indexMultipleDimsOnly));

        // Index must stay within bounds
        for (size_t i = 0; i < idx.size(); i++) {
            assert(str_assert(idx[i] < shape.view[i], AssertMessage::indexOutOfBounds));
        }

        TensorIndexer indexer(this);
        return indexer.at(idx);
    }

    bool Tensor::isContiguous() const {
        return std::ranges::all_of(shape.ranges.begin(), shape.ranges.end(),
                                   [](const Range &range) { return range.step > 1; });
    }

    Tensor &Tensor::at(const std::vector<size_t> &idx) {
        auto outTensor = atHelper(idx);
        return *outTensor;
    }

    Tensor &Tensor::at(const std::vector<Range> &ranges) {
        TensorIndexer indexer(this);
        return *indexer.at(ranges);
    }

    Tensor &Tensor::arange(const Shape &shape, real start, real step) {
        auto outTensor = new Tensor(shape);
        auto outOp = new ArangeOp(outTensor, start, step);
        outOp->forward();
        Graph::addOp(outOp);
        return *outTensor;
    }

    Tensor &Tensor::randint(const Shape &shape, int min, int max) {
        auto outTensor = new Tensor(shape);
        auto outOp = new RandintOp(outTensor, min, max);
        outOp->forward();
        Graph::addOp(outOp);
        return *outTensor;
    }

    Tensor &Tensor::randn(const Shape &shape) {
        auto outTensor = new Tensor(shape);
        auto outOp = new RandnOp(outTensor);
        outOp->forward();
        Graph::addOp(outOp);
        return *outTensor;
    }

    Tensor &Tensor::fromConst(const Shape &shape, real c) {
        auto outTensor = new Tensor(shape);
        auto outOp = new ConstOp(outTensor, c);
        outOp->forward();
        Graph::addOp(outOp);
        return *outTensor;
    }

    Tensor &Tensor::fromArr(const Shape &shape, const real *data) {
        auto outTensor = new Tensor(shape);
        auto outOp = new FromArrOp(outTensor, data);
        outOp->forward();
        Graph::addOp(outOp);
        return *outTensor;
    }

    Tensor &Tensor::operator[](size_t idx) {
        auto outTensor = atHelper(std::vector(1, idx));
        return *outTensor;
    }

    Tensor &Tensor::operator+(Tensor &rhs) {
        assert(str_assert(shape == rhs.shape, AssertMessage::shapesMismatched));
        auto outTensor = new Tensor(shape);
        auto outOp = new AddOp(op, rhs.op, outTensor);
        outOp->forward();
        Graph::addOp(outOp);
        return *outTensor;
    }

    Tensor &Tensor::operator+(real c) {
        auto &rhs = fromConst(shape, c);
        return *this + rhs;
    }

    Tensor &Tensor::operator-(Tensor &rhs) {
        assert(str_assert(shape == rhs.shape, AssertMessage::shapesMismatched));
        auto outTensor = new Tensor(shape);
        auto outOp = new SubOp(op, rhs.op, outTensor);
        outOp->forward();
        Graph::addOp(outOp);
        return *outTensor;
    }

    Tensor &Tensor::operator-(real c) {
        auto &rhs = fromConst(shape, c);
        return *this - rhs;
    }

    Tensor &Tensor::operator*(Tensor &rhs) {
        assert(str_assert(shape == rhs.shape, AssertMessage::shapesMismatched));
        auto outTensor = new Tensor(shape);
        auto outOp = new MulOp(op, rhs.op, outTensor);
        outOp->forward();
        Graph::addOp(outOp);
        return *outTensor;
    }

    Tensor &Tensor::operator*(real c) {
        auto &rhs = fromConst(shape, c);
        return *this * rhs;
    }

    Tensor &Tensor::operator/(Tensor &rhs) {
        assert(str_assert(shape == rhs.shape, AssertMessage::shapesMismatched));
        auto outTensor = new Tensor(shape);
        auto outOp = new DivOp(op, rhs.op, outTensor);
        outOp->forward();
        Graph::addOp(outOp);
        return *outTensor;
    }

    Tensor &Tensor::operator/(real c) {
        auto &rhs = fromConst(shape, c);
        return *this / rhs;
    }

    Tensor &Tensor::exp() {
        auto outTensor = new Tensor(shape);
        auto outOp = new ExpOp(op, outTensor);
        outOp->forward();
        Graph::addOp(outOp);
        return *outTensor;
    }

    Tensor &Tensor::eq(Tensor &rhs) {
        assert(str_assert(shape == rhs.shape, AssertMessage::shapesMismatched));
        auto outTensor = new Tensor(shape);
        auto outOp = new EqOp(op, rhs.op, outTensor);
        outOp->forward();
        Graph::addOp(outOp);
        return *outTensor;
    }

    Tensor &Tensor::eq(real c) {
        auto &rhs = fromConst(shape, c);
        return eq(rhs);
    }

    bool Tensor::operator==(Tensor &rhs) {
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

    Tensor &Tensor::neq(Tensor &rhs) {
        assert(str_assert(shape == rhs.shape, AssertMessage::shapesMismatched));
        auto outTensor = new Tensor(shape);
        auto outOp = new NeqOp(op, rhs.op, outTensor);
        outOp->forward();
        Graph::addOp(outOp);
        return *outTensor;
    }

    Tensor &Tensor::neq(real c) {
        auto &rhs = fromConst(shape, c);
        return neq(rhs);
    }

    bool Tensor::operator!=(Tensor &rhs) {
        return !(*this == rhs);
    }

    Tensor &Tensor::operator<(Tensor &rhs) {
        assert(str_assert(shape == rhs.shape, AssertMessage::shapesMismatched));
        auto outTensor = new Tensor(shape);
        auto outOp = new LessOp(op, rhs.op, outTensor);
        outOp->forward();
        Graph::addOp(outOp);
        return *outTensor;
    }

    Tensor &Tensor::operator<(real c) {
        auto &rhs = fromConst(shape, c);
        return *this < rhs;
    }

    Tensor &Tensor::operator>(Tensor &rhs) {
        assert(str_assert(shape == rhs.shape, AssertMessage::shapesMismatched));
        auto outTensor = new Tensor(shape);
        auto outOp = new GreaterOp(op, rhs.op, outTensor);
        outOp->forward();
        Graph::addOp(outOp);
        return *outTensor;
    }

    Tensor &Tensor::operator>(real c) {
        auto &rhs = fromConst(shape, c);
        return *this > rhs;
    }

    Tensor &Tensor::operator<=(Tensor &rhs) {
        assert(str_assert(shape == rhs.shape, AssertMessage::shapesMismatched));
        auto outTensor = new Tensor(shape);
        auto outOp = new LeqOp(op, rhs.op, outTensor);
        outOp->forward();
        Graph::addOp(outOp);
        return *outTensor;
    }

    Tensor &Tensor::operator<=(real c) {
        auto &rhs = fromConst(shape, c);
        return *this <= rhs;
    }

    Tensor &Tensor::operator>=(Tensor &rhs) {
        assert(str_assert(shape == rhs.shape, AssertMessage::shapesMismatched));
        auto outTensor = new Tensor(shape);
        auto outOp = new GeqOp(op, rhs.op, outTensor);
        outOp->forward();
        Graph::addOp(outOp);
        return *outTensor;
    }

    Tensor &Tensor::operator>=(real c) {
        auto &rhs = fromConst(shape, c);
        return *this >= rhs;
    }

    Tensor &Tensor::operator=(const Tensor &rhs) {
        if (this == &rhs) {
            return *this;
        }

        assert(str_assert(shape == rhs.shape, AssertMessage::shapesMismatched));

        if (vec->refCount > 1) {
            vec->refCount--;
        } else {
            delete vec;
        }

        vec = rhs.vec;
        vec->refCount++;
        // Keep the edges the same
        // Do not copy op since that creates new unwanted edges
        return *this;
    }

    Tensor &Tensor::operator=(real c) {
        auto &constTensor = fromConst(shape, c);
        return (*this = constTensor);
    }

    Tensor &Tensor::operator+=(Tensor &rhs) {
        assert(str_assert(shape == rhs.shape, AssertMessage::shapesMismatched));
        auto outOp = new AddAssignOp(rhs.op, this);
        outOp->forward();
        Graph::addOp(outOp);
        return *this;
    }

    Tensor &Tensor::operator+=(real c) {
        auto &constTensor = fromConst(shape, c);
        return *this += constTensor;
    }

    Tensor &Tensor::operator-=(Tensor &rhs) {
        assert(str_assert(shape == rhs.shape, AssertMessage::shapesMismatched));
        auto outOp = new SubAssignOp(rhs.op, this);
        outOp->forward();
        Graph::addOp(outOp);
        return *this;
    }

    Tensor &Tensor::operator-=(real c) {
        auto &constTensor = fromConst(shape, c);
        return *this -= constTensor;
    }

    Tensor &Tensor::operator*=(Tensor &rhs) {
        assert(str_assert(shape == rhs.shape, AssertMessage::shapesMismatched));
        auto outOp = new MulAssignOp(rhs.op, this);
        outOp->forward();
        Graph::addOp(outOp);
        return *this;
    }

    Tensor &Tensor::operator*=(real c) {
        auto &constTensor = fromConst(shape, c);
        return *this *= constTensor;
    }

    Tensor &Tensor::operator/=(Tensor &rhs) {
        assert(str_assert(shape == rhs.shape, AssertMessage::shapesMismatched));
        auto outOp = new DivAssignOp(rhs.op, this);
        outOp->forward();
        Graph::addOp(outOp);
        return *this;
    }

    Tensor &Tensor::operator/=(real c) {
        auto &constTensor = fromConst(shape, c);
        return *this /= constTensor;
    }

    Tensor &operator+(real c, Tensor &rhs) {
        auto &constTensor = Tensor::fromConst(rhs.shape, c);
        return constTensor + rhs;
    }

    Tensor &operator-(real c, Tensor &rhs) {
        auto &constTensor = Tensor::fromConst(rhs.shape, c);
        return constTensor - rhs;
    }

    Tensor &operator*(real c, Tensor &rhs) {
        auto &constTensor = Tensor::fromConst(rhs.shape, c);
        return constTensor * rhs;
    }

    Tensor &operator/(real c, Tensor &rhs) {
        auto &constTensor = Tensor::fromConst(rhs.shape, c);
        return constTensor / rhs;
    }

    Tensor &Tensor::relu() {
        auto outTensor = new Tensor(shape);
        auto outOp = new ReluOp(op, outTensor);
        outOp->forward();
        Graph::addOp(outOp);
        return *outTensor;
    }

    Tensor &Tensor::sum() {
        auto outTensor = new Tensor(Shape());
        auto outOp = new SumOp(op, outTensor);
        outOp->forward();
        Graph::addOp(outOp);
        return *outTensor;
    }

    void Tensor::backward() {
    }
}
