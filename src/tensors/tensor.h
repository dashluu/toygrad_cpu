#pragma once

#include "shape.h"
#include "vec.h"
#include "assert/str_assert.h"

namespace Toygrad::Tensor {
    class Tensor final : public std::enable_shared_from_this<Tensor> {
        Shape shape;
        std::shared_ptr<Vec> vec = nullptr;
        static size_t idCounter;
        size_t id{};
        std::vector<Op *> ops = std::vector<Op *>();
        TensorPtr grad;
        std::vector<Tensor *> edges = std::vector<Tensor *>();
        // TensorGraph is incomplete so raw pointer is used
        TensorGraph *graph = nullptr;

        friend class NN::Module;
        friend class TensorGraph;
        friend class TensorDraw;
        friend struct Op;
        friend struct LeafOp;
        friend struct UnOp;
        friend struct BinOp;
        friend struct ConstOp;
        friend struct IndexOp;
        friend struct ArangeOp;
        friend struct RandintOp;
        friend struct RandnOp;
        friend struct FromArrOp;
        friend struct SumOp;
        friend struct AddOp;
        friend struct AddAssignOp;
        friend struct SubOp;
        friend struct SubAssignOp;
        friend struct MulOp;
        friend struct MulAssignOp;
        friend struct DivOp;
        friend struct DivAssignOp;
        friend struct PowOp;
        friend struct LogOp;
        friend struct SinOp;
        friend struct CosOp;
        friend struct ExpOp;
        friend struct RecipOp;
        friend struct NegOp;
        friend struct SqOp;
        friend struct SqrtOp;
        friend struct AliasOp;
        friend struct DiffAliasOp;
        friend struct EqOp;
        friend struct NeqOp;
        friend struct LessOp;
        friend struct GreaterOp;
        friend struct LeqOp;
        friend struct GeqOp;
        friend struct MaxOp;
        friend struct MinOp;
        friend struct PermOp;
        friend struct ReluOp;
        friend struct SigmoidOp;
        friend struct SoftmaxOp;
        friend struct CopyOp;
        friend struct MatmulOp;

        Tensor();

        explicit Tensor(const std::vector<size_t> &view, bool initStrides = true): Tensor(Shape(view), initStrides) {
        }

        TensorPtr getThis() { return shared_from_this(); }

        inline void clearOps();

        static inline void realizeOp(Op *op, bool lazy);

        static TensorPtr initTensor(const Shape &shape, bool initStrides, TensorPtr outTensor) {
            if (outTensor != nullptr) {
                assert(Error::str_assert(outTensor->shape == shape,
                    Error::Message::shapesMismatched("tensor modification", outTensor->shape, shape)));
                return outTensor;
            }

            return std::make_shared<Tensor>(shape, initStrides);
        }

        TensorPtr index(const std::vector<size_t> &indices, bool lazy = true, TensorPtr outTensor = nullptr);

        TensorPtr index(const std::vector<Range> &ranges, bool lazy = true, TensorPtr outTensor = nullptr);

        void initVec() {
            if (vec == nullptr) {
                vec = std::make_shared<Vec>(shape.getSize());
            }
        }

        void initVec(real c) {
            if (vec == nullptr) {
                vec = std::make_shared<Vec>(shape.getSize(), c);
            }
        }

        void initGrad() {
            if (grad == nullptr) {
                grad = initTensor(shape, false, nullptr);
                grad->initVec();
            }
        }

        void initGrad(real c) {
            if (grad == nullptr) {
                grad = initTensor(shape, false, nullptr);
                grad->initVec(c);
            }
        }

        bool isDimValid(int64_t dim) const { return dim >= -1 && dim < static_cast<int>(shape.getNumDims()); }

        TensorPtr perm(const Shape &target, bool lazy = true, TensorPtr outTensor = nullptr);

        TensorPtr alias(const Shape &target, bool lazy = true, TensorPtr outTensor = nullptr);

    public:
        explicit Tensor(const Shape &shape, bool initStrides = true);

        Tensor(const Tensor &tensor) = delete;

        ~Tensor();

        /**
         * Gets the ID of the tensor.
         * @return the tensor's ID.
         */
        size_t getId() const { return id; }

        /**
         * Gets the shape of the tensor.
         * @return the shape of the current tensor.
         */
        const Shape &getShape() const { return shape; }

        /**
         * Gets the gradient of the current tensor.
         * @return the gradient tensor of the current tensor.
         */
        TensorPtr getGrad() const { return grad; }

        /**
         * Gets a pointer to the underlying memory.
         * @return a pointer to the underlying memory.
         */
        std::shared_ptr<Vec> getVec() const { return vec; }

        /**
         * Checks if the tensor's memory is contiguous.
         * @return true if the underlying memory is accessed contiguously and false otherwise.
         */
        bool isContiguous() const;

        /**
         * Checks if the tensor is broadcastable to a given shape.
         * @param target the target shape to be broadcasted to.
         * @return true if the tensor can be broadcasted and false otherwise.
         */
        bool isBroadcastableTo(const Shape &target) const;

        /**
         * Checks if the tensor is broadcastable to a given shape.
         * @param view the view of the target shape to be broadcasted to.
         * @return true if the tensor can be broadcasted and false otherwise.
         */
        bool isBroadcastableTo(const std::vector<size_t> &view) const {
            return isBroadcastableTo(Shape(view));
        }

        /**
         * Broadcasts the tensor to a given shape.
         * @param target the target shape to be broadcasted to.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr broadcastTo(const Shape &target, bool lazy = true, TensorPtr outTensor = nullptr);

        /**
         * Broadcasts the tensor to a given shape.
         * @param view the view of the target shape to be broadcasted to.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr broadcastTo(const std::vector<size_t> &view, bool lazy = true, TensorPtr outTensor = nullptr) {
            return broadcastTo(Shape(view), lazy, std::move(outTensor));
        }

        /**
         * Creates a shallow copy of the tensor using the same underlying memory.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return a shallow copy of the tensor.
         */
        TensorPtr alias(bool lazy = true, TensorPtr outTensor = nullptr) {
            return alias(shape, lazy, std::move(outTensor));
        }

        /**
         * Creates a shallow copy of the tensor that uses the same underlying memory, has the same shape and is differentiable.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return a shallow copy of the tensor with the same shape is differentiable.
         */
        TensorPtr diffAlias(bool lazy = true, TensorPtr outTensor = nullptr);

        /**
         * Creates a deep copy of the tensor using the same underlying memory but without any graph connection.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return a deep copy of the tensor.
         */
        TensorPtr copy(bool lazy = true, TensorPtr outTensor = nullptr);

        /**
         * Squeezes the tensor in a given dimension.
         * @param dim the dimension to squeeze the tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr squeeze(int64_t dim = -1, bool lazy = true, TensorPtr outTensor = nullptr);

        /**
         * Inserts(before) a dimension of size one at a position.
         * @param dim the dimension at which dimension of size one to be inserted. If dim is -1, it inserts the new
         * dimension at the end.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return a new tensor with a dimension of size one inserted at the specified position.
         */
        TensorPtr unsqueeze(int64_t dim = -1, bool lazy = true, TensorPtr outTensor = nullptr);

        /**
         * Accesses tensor data at a given index.
         * @param idx the index to access the tensor at.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result as a tensor.
         */
        TensorPtr at(size_t idx, bool lazy = true, TensorPtr outTensor = nullptr) {
            return index({idx}, lazy, std::move(outTensor));
        }

        /**
         * Accesses tensor data at given indices.
         * @param indices the indices to access the tensor at.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result as a tensor.
         */
        TensorPtr at(const std::vector<size_t> &indices, bool lazy = true, TensorPtr outTensor = nullptr) {
            return index(indices, lazy, std::move(outTensor));
        }

        /**
         * Accesses tensor data within given ranges.
         * @param ranges the ranges for accessing the tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result as a tensor.
         */
        TensorPtr at(const std::vector<Range> &ranges, bool lazy = true, TensorPtr outTensor = nullptr);

        /**
         * Creates a new tensor containing increasing integers.
         * @param shape the shape of the tensor.
         * @param start the starting integer.
         * @param step the step to increment each integer.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the newly created tensor.
         */
        static TensorPtr arange(const Shape &shape, real start, real step = 1., bool lazy = true,
                                TensorPtr outTensor = nullptr);

        /**
         * Creates a new tensor containing increasing integers.
         * @param view the view of the shape of the tensor.
         * @param start the starting integer.
         * @param step the step to increment each integer.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the newly created tensor.
         */
        static TensorPtr arange(const std::vector<size_t> &view, real start, real step = 1., bool lazy = true,
                                TensorPtr outTensor = nullptr) {
            return arange(Shape(view), start, step, lazy, std::move(outTensor));
        }

        /**
         * Creates a new tensor containing random integers in the range[min, max].
         * @param shape the shape of the tensor.
         * @param min the lower-bound integer.
         * @param max the upper-bound integer.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the newly created tensor.
         */
        static TensorPtr randint(const Shape &shape, int64_t min, int64_t max, bool lazy = true,
                                 TensorPtr outTensor = nullptr);

        /**
         * Creates a new tensor containing random integers in the range[min, max].
         * @param view the view of the shape of the tensor.
         * @param min the lower-bound integer.
         * @param max the upper-bound integer.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the newly created tensor.
         */
        static TensorPtr randint(const std::vector<size_t> &view, int64_t min, int64_t max, bool lazy = true,
                                 TensorPtr outTensor = nullptr) {
            return randint(Shape(view), min, max, lazy, std::move(outTensor));
        }

        /**
         * Creates a new tensor containing random values using normal distribution.
         * @param shape the shape of the tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the newly created tensor.
         */
        static TensorPtr randn(const Shape &shape, bool lazy = true, TensorPtr outTensor = nullptr);

        /**
         * Creates a new tensor containing random values using normal distribution.
         * @param view the view of the shape of the tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the newly created tensor.
         */
        static TensorPtr randn(const std::vector<size_t> &view, bool lazy = true, TensorPtr outTensor = nullptr) {
            return randn(Shape(view), lazy, std::move(outTensor));
        }

        /**
         * Creates a new tensor containing the same constant value.
         * @param shape the shape of the tensor.
         * @param c the constant value.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the newly created tensor.
         */
        static TensorPtr fromConst(const Shape &shape, real c, bool lazy = true, TensorPtr outTensor = nullptr);

        /**
         * Creates a new tensor containing the same constant value.
         * @param view the view of the shape of the tensor.
         * @param c the constant value.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the newly created tensor.
         */
        static TensorPtr fromConst(const std::vector<size_t> &view, real c, bool lazy = true,
                                   TensorPtr outTensor = nullptr) {
            return fromConst(Shape(view), c, lazy, std::move(outTensor));
        }

        /**
         * Creates a new tensor containing 0s.
         * @param shape the shape of the tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the newly created tensor.
         */
        static TensorPtr zeros(const Shape &shape, bool lazy = true, TensorPtr outTensor = nullptr) {
            return fromConst(shape, 0.0, lazy, std::move(outTensor));
        }

        /**
         * Creates a new tensor containing 0s.
         * @param view the view of the shape of the tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the newly created tensor.
         */
        static TensorPtr zeros(const std::vector<size_t> &view, bool lazy = true, TensorPtr outTensor = nullptr) {
            return fromConst(view, 0.0, lazy, std::move(outTensor));
        }

        /**
         * Creates a new tensor with 0s whose shape is the same as the given tensor.
         * @param tensor the tensor whose shape is the same as that of the new tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the newly created tensor.
         */
        static TensorPtr zerosLike(const TensorPtr &tensor, bool lazy = true, TensorPtr outTensor = nullptr) {
            return zeros(tensor->shape, lazy, std::move(outTensor));
        }

        /**
         * Creates a new tensor with 0s whose shape is the same as the given tensor.
         * @param tensor the tensor whose shape is the same as that of the new tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the newly created tensor.
         */
        static TensorPtr zerosLike(const Tensor &tensor, bool lazy = true, TensorPtr outTensor = nullptr) {
            return zeros(tensor.shape, lazy, std::move(outTensor));
        }

        /**
         * Creates a new tensor containing 1s.
         * @param shape the shape of the tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the newly created tensor.
         */
        static TensorPtr ones(const Shape &shape, bool lazy = true, TensorPtr outTensor = nullptr) {
            return fromConst(shape, 1.0, lazy, std::move(outTensor));
        }

        /**
         * Creates a new tensor containing 1s.
         * @param view the view of the shape of the tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the newly created tensor.
         */
        static TensorPtr ones(const std::vector<size_t> &view, bool lazy = true, TensorPtr outTensor = nullptr) {
            return fromConst(view, 1.0, lazy, std::move(outTensor));
        }

        /**
         * Creates a new tensor with 1s whose shape is the same as the given tensor.
         * @param tensor the tensor whose shape is the same as that of the new tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the newly created tensor.
         */
        static TensorPtr onesLike(const TensorPtr &tensor, bool lazy = true, TensorPtr outTensor = nullptr) {
            return ones(tensor->shape, lazy, std::move(outTensor));
        }

        /**
         * Creates a new tensor with 1s whose shape is the same as the given tensor.
         * @param tensor the tensor whose shape is the same as that of the new tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the newly created tensor.
         */
        static TensorPtr onesLike(const Tensor &tensor, bool lazy = true, TensorPtr outTensor = nullptr) {
            return ones(tensor.shape, lazy, std::move(outTensor));
        }

        /**
         * Constructs a tensor given a shape and an array.
         * @param shape the tensor shape.
         * @param data the array containing the tensor data.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return a new tensor with the given shape and data.
         */
        static TensorPtr fromArr(const Shape &shape, const real *data, bool lazy = true, TensorPtr outTensor = nullptr);

        /**
         * Constructs a tensor given a shape and an array.
         * @param view the view of the tensor shape.
         * @param data the array containing the tensor data.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return a new tensor with the given shape and data.
         */
        static TensorPtr fromArr(const std::vector<size_t> &view, const real *data, bool lazy = true,
                                 TensorPtr outTensor = nullptr) {
            return fromArr(Shape(view), data, lazy, std::move(outTensor));
        }

        /**
         * Constructs a tensor given a shape and a vector.
         * @param shape the tensor shape.
         * @param data the vector containing the tensor data.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return a new tensor with the given shape and data.
         */
        static TensorPtr fromVec(const Shape &shape, const std::vector<real> &data, bool lazy = true,
                                 TensorPtr outTensor = nullptr) {
            return fromArr(shape, data.data(), lazy, std::move(outTensor));
        }

        /**
         * Constructs a tensor given a shape and a vector.
         * @param view the view of the tensor shape.
         * @param data the vector containing the tensor data.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return a new tensor with the given shape and data.
         */
        static TensorPtr fromVec(const std::vector<size_t> &view, const std::vector<real> &data, bool lazy = true,
                                 TensorPtr outTensor = nullptr) {
            return fromVec(Shape(view), data, lazy, std::move(outTensor));
        }

        friend std::ostream &operator<<(std::ostream &stream, const Tensor &tensor);

        TensorPtr operator[](size_t idx);

        /**
         * Adds two tensors element-wise.
         * @param rhs the right tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr add(const TensorPtr &rhs, bool lazy = true, TensorPtr outTensor = nullptr) {
            return add(*rhs, lazy, std::move(outTensor));
        }

        /**
         * Adds two tensors element-wise.
         * @param rhs the right tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr add(Tensor &rhs, bool lazy = true, TensorPtr outTensor = nullptr);

        /**
         * Adds a constant to each element in the tensor.
         * @param c the constant to be added.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr add(real c, bool lazy = true, TensorPtr outTensor = nullptr) {
            return add(fromConst(shape, c, lazy, nullptr), lazy, std::move(outTensor));
        }

        /**
         * Subtracts two tensors element-wise.
         * @param rhs the right tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr sub(const TensorPtr &rhs, bool lazy = true, TensorPtr outTensor = nullptr) {
            return sub(*rhs, lazy, std::move(outTensor));
        }

        /**
         * Subtracts two tensors element-wise.
         * @param rhs the right tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr sub(Tensor &rhs, bool lazy = true, TensorPtr outTensor = nullptr);

        /**
         * Subtracts a constant from each element in the tensor.
         * @param c the constant to be subtracted.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr sub(real c, bool lazy = true, TensorPtr outTensor = nullptr) {
            return sub(fromConst(shape, c, lazy, nullptr), lazy, std::move(outTensor));
        }

        /**
         * Multiplies two tensors element-wise.
         * @param rhs the right tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr mul(const TensorPtr &rhs, bool lazy = true, TensorPtr outTensor = nullptr) {
            return mul(*rhs, lazy, std::move(outTensor));
        }

        /**
         * Multiplies two tensors element-wise.
         * @param rhs the right tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr mul(Tensor &rhs, bool lazy = true, TensorPtr outTensor = nullptr);

        /**
         * Multiplies each element by a constant in the tensor.
         * @param c the constant to be multiplied.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr mul(real c, bool lazy = true, TensorPtr outTensor = nullptr) {
            return mul(fromConst(shape, c, lazy, nullptr), lazy, std::move(outTensor));
        }

        /**
         * Divides two tensors element-wise.
         * @param rhs the right tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr div(const TensorPtr &rhs, bool lazy = true, TensorPtr outTensor = nullptr) {
            return div(*rhs, lazy, std::move(outTensor));
        }

        /**
         * Divides two tensors element-wise.
         * @param rhs the right tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr div(Tensor &rhs, bool lazy = true, TensorPtr outTensor = nullptr);

        /**
         * Divides each element by a constant in the tensor.
         * @param c the constant to be divided.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr div(real c, bool lazy = true, TensorPtr outTensor = nullptr) {
            return div(fromConst(shape, c, lazy, nullptr), lazy, std::move(outTensor));
        }

        /**
         * Raises each element in the tensor by a given power.
         * @param c the power to be raised.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr pow(real c, bool lazy = true, TensorPtr outTensor = nullptr);

        /**
         * Computes the natural logarithm of each element in the tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr log(bool lazy = true, TensorPtr outTensor = nullptr);

        /**
         * Computes the sine of each element in the tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr sin(bool lazy = true, TensorPtr outTensor = nullptr);

        /**
         * Computes the cosine of each element in the tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr cos(bool lazy = true, TensorPtr outTensor = nullptr);

        /**
         * Computes the narutal exponent of each element in the tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr exp(bool lazy = true, TensorPtr outTensor = nullptr);

        /**
         * Computes the reciprocal of each element in the tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr recip(real c = 1, bool lazy = true, TensorPtr outTensor = nullptr);

        /**
         * Computes the square of each element in the tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr sq(bool lazy = true, TensorPtr outTensor = nullptr);

        /**
         * Computes the square root of each element in the tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr sqrt(bool lazy = true, TensorPtr outTensor = nullptr);

        /**
         * Computes the negation of each element in the tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr neg(bool lazy = true, TensorPtr outTensor = nullptr);

        /**
         * Checks if two tensors are equal elementwise.
         * @param rhs the right tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr eq(const TensorPtr &rhs, bool lazy = true, TensorPtr outTensor = nullptr) {
            return eq(*rhs, lazy, std::move(outTensor));
        }

        /**
         * Checks if two tensors are equal elementwise.
         * @param rhs the right tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr eq(Tensor &rhs, bool lazy = true, TensorPtr outTensor = nullptr);

        /**
         * Checks if each element in the tensor is equal to a constant.
         * @param c the constant to be compared.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr eq(real c, bool lazy = true, TensorPtr outTensor = nullptr) {
            return eq(fromConst(shape, c, lazy, nullptr), lazy, std::move(outTensor));
        }

        /**
         * Checks if tensors are equal elementwise.
         * @param rhs the right tensor.
         * @return true if all elements are equal and false otherwise.
         */
        bool operator==(const Tensor &rhs) const;

        /**
         * Checks if two tensors are not equal elementwise.
         * @param rhs the right tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr neq(const TensorPtr &rhs, bool lazy = true, TensorPtr outTensor = nullptr) {
            return neq(*rhs, lazy, std::move(outTensor));
        }

        /**
         * Checks if two tensors are not equal elementwise.
         * @param rhs the right tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr neq(Tensor &rhs, bool lazy = true, TensorPtr outTensor = nullptr);

        /**
         * Checks if each element in the tensor is not equal to a constant.
         * @param c the constant to be compared.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr neq(real c, bool lazy = true, TensorPtr outTensor = nullptr) {
            return neq(fromConst(shape, c, lazy, nullptr), lazy, std::move(outTensor));
        }

        /**
         * Checks if tensors are not equal elementwise.
         * @param rhs the right tensor.
         * @return true if one pair of elements are not equal and false otherwise.
         */
        bool operator!=(const Tensor &rhs) const;

        /**
         * Checks if the left tensor is less than the right tensor elementwise.
         * @param rhs the right tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr lt(const TensorPtr &rhs, bool lazy = true, TensorPtr outTensor = nullptr) {
            return lt(*rhs, lazy, std::move(outTensor));
        }

        /**
         * Checks if the left tensor is less than the right tensor elementwise.
         * @param rhs the right tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr lt(Tensor &rhs, bool lazy = true, TensorPtr outTensor = nullptr);

        /**
         * Checks if each element of the tensor is less than a constant.
         * @param c the constant to be compared.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr lt(real c, bool lazy = true, TensorPtr outTensor = nullptr) {
            return lt(fromConst(shape, c, lazy, nullptr), lazy, std::move(outTensor));
        }

        /**
         * Checks if the left tensor is greater than the right tensor elementwise.
         * @param rhs the right tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr gt(const TensorPtr &rhs, bool lazy = true, TensorPtr outTensor = nullptr) {
            return gt(*rhs, lazy, std::move(outTensor));
        }

        /**
         * Checks if the left tensor is greater than the right tensor elementwise.
         * @param rhs the right tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr gt(Tensor &rhs, bool lazy = true, TensorPtr outTensor = nullptr);

        /**
         * Checks if each element of the tensor is greater than a constant.
         * @param c the constant to be compared.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr gt(real c, bool lazy = true, TensorPtr outTensor = nullptr) {
            return gt(fromConst(shape, c, lazy, nullptr), lazy, std::move(outTensor));
        }

        /**
         * Checks if the left tensor is less than or equal to the right tensor elementwise.
         * @param rhs the right tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr leq(const TensorPtr &rhs, bool lazy = true, TensorPtr outTensor = nullptr) {
            return leq(*rhs, lazy, std::move(outTensor));
        }

        /**
         * Checks if the left tensor is less than or equal to the right tensor elementwise.
         * @param rhs the right tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr leq(Tensor &rhs, bool lazy = true, TensorPtr outTensor = nullptr);

        /**
         * Checks if each element of the tensor is less than or equal to a constant.
         * @param c the constant to be compared.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr leq(real c, bool lazy = true, TensorPtr outTensor = nullptr) {
            return leq(fromConst(shape, c, lazy, nullptr), lazy, std::move(outTensor));
        }

        /**
         * Checks if the left tensor is greater than or equal to the right tensor elementwise.
         * @param rhs the right tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr geq(const TensorPtr &rhs, bool lazy = true, TensorPtr outTensor = nullptr) {
            return geq(*rhs, lazy, std::move(outTensor));
        }

        /**
         * Checks if the left tensor is greater than or equal to the right tensor elementwise.
         * @param rhs the right tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr geq(Tensor &rhs, bool lazy = true, TensorPtr outTensor = nullptr);

        /**
         * Checks if each element of the tensor is greater than or equal to a constant.
         * @param c the constant to be compared.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr geq(real c, bool lazy = true, TensorPtr outTensor = nullptr) {
            return geq(fromConst(shape, c, lazy, nullptr), lazy, std::move(outTensor));
        }

        Tensor &operator=(const Tensor &rhs) = delete;

        /**
         * Increments each element in the current tensor by the corresponding element in the right tensor.
         * @param rhs the right tensor.
         * @param lazy whether the operation is executed lazily.
         * @return the result tensor, the same as the current tensor.
         */
        TensorPtr addAssign(const TensorPtr &rhs, bool lazy = true) { return addAssign(*rhs, lazy); }

        /**
         * Increments each element in the current tensor by the corresponding element in the right tensor.
         * @param rhs the right tensor.
         * @param lazy whether the operation is executed lazily.
         * @return the result tensor, the same as the current tensor.
         */
        TensorPtr addAssign(Tensor &rhs, bool lazy = true);

        /**
         * Increments each element in the current tensor by a constant.
         * @param c the constant to be incremented by.
         * @param lazy whether the operation is executed lazily.
         * @return the result tensor, the same as the current tensor.
         */
        TensorPtr addAssign(real c, bool lazy = true) {
            return addAssign(fromConst(shape, c, lazy, nullptr), lazy);
        }

        /**
         * Decrements each element in the current tensor by the corresponding element in the right tensor.
         * @param rhs the right tensor.
         * @param lazy whether the operation is executed lazily.
         * @return the result tensor, the same as the current tensor.
         */
        TensorPtr subAssign(const TensorPtr &rhs, bool lazy = true) { return subAssign(*rhs, lazy); }

        /**
         * Decrements each element in the current tensor by the corresponding element in the right tensor.
         * @param rhs the right tensor.
         * @param lazy whether the operation is executed lazily.
         * @return the result tensor, the same as the current tensor.
         */
        TensorPtr subAssign(Tensor &rhs, bool lazy = true);

        /**
         * Decrements each element in the current tensor by a constant.
         * @param c the constant to be decremented by.
         * @param lazy whether the operation is executed lazily.
         * @return the result tensor, the same as the current tensor.
         */
        TensorPtr subAssign(real c, bool lazy = true) {
            return subAssign(fromConst(shape, c, lazy, nullptr), lazy);
        }

        /**
         * Multiplies in-place each element in the current tensor by the corresponding element in the right tensor.
         * @param rhs the right tensor.
         * @param lazy whether the operation is executed lazily.
         * @return the result tensor, the same as the current tensor.
         */
        TensorPtr mulAssign(const TensorPtr &rhs, bool lazy = true) { return mulAssign(*rhs, lazy); }

        /**
         * Multiplies in-place each element in the current tensor by the corresponding element in the right tensor.
         * @param rhs the right tensor.
         * @param lazy whether the operation is executed lazily.
         * @return the result tensor, the same as the current tensor.
         */
        TensorPtr mulAssign(Tensor &rhs, bool lazy = true);

        /**
         * Multiplies in-place each element in the current tensor by a constant.
         * @param c the constant to be multiplied by.
         * @param lazy whether the operation is executed lazily.
         * @return the result tensor, the same as the current tensor.
         */
        TensorPtr mulAssign(real c, bool lazy = true) {
            return mulAssign(fromConst(shape, c, lazy, nullptr), lazy);
        }

        /**
         * Divides in-place each element in the current tensor by the corresponding element in the right tensor.
         * @param rhs the right tensor.
         * @param lazy whether the operation is executed lazily.
         * @return the result tensor, the same as the current tensor.
         */
        TensorPtr divAssign(const TensorPtr &rhs, bool lazy = true) { return divAssign(*rhs, lazy); }

        /**
         * Divides in-place each element in the current tensor by the corresponding element in the right tensor.
         * @param rhs the right tensor.
         * @param lazy whether the operation is executed lazily.
         * @return the result tensor, the same as the current tensor.
         */
        TensorPtr divAssign(Tensor &rhs, bool lazy = true);

        /**
         * Divides in-place each element in the current tensor by a constant.
         * @param c the constant to be divided by.
         * @param lazy whether the operation is executed lazily.
         * @return the result tensor, the same as the current tensor.
         */
        TensorPtr divAssign(real c, bool lazy = true) {
            return divAssign(fromConst(shape, c, lazy, nullptr), lazy);
        }

        /**
         * Computes Rectified Linear Unit(ReLU) elementwise in the tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr relu(bool lazy = true, TensorPtr outTensor = nullptr);

        /**
         * Computes Sigmoid elementwise in the tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr sigmoid(bool lazy = true, TensorPtr outTensor = nullptr);

        /**
         * Computes Softmax in a given tensor dimension.
         * @param dim the dimension to be computed in.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr softmax(int64_t dim = -1, bool lazy = true, TensorPtr outTensor = nullptr);

        /**
         * Matrix multiplies two tensors in the last two dimensions.
         * @param rhs the right tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr matmul(const TensorPtr &rhs, bool lazy = true, TensorPtr outTensor = nullptr) {
            return matmul(*rhs, lazy, std::move(outTensor));
        }

        /**
         * Matrix multiplies two tensors in the last two dimensions.
         * @param rhs the right tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr matmul(Tensor &rhs, bool lazy = true, TensorPtr outTensor = nullptr);

        /**
         * Reshapes the tensor to a given shape.
         * @param target the target shape to be reshaped to.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr reshape(const Shape &target, bool lazy = true, TensorPtr outTensor = nullptr);

        /**
         * Reshapes the tensor to a given shape.
         * @param view the view of the target shape to be reshaped to.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr reshape(const std::vector<size_t> &view, bool lazy = true, TensorPtr outTensor = nullptr) {
            return reshape(Shape(view), lazy, std::move(outTensor));
        }

        /**
         * Flattens the tensor.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr flatten(bool lazy = true, TensorPtr outTensor = nullptr) {
            return reshape({shape.getSize()}, lazy, std::move(outTensor));
        }

        /**
         * Computes the summation in a given tensor dimension.
         * @param dim the dimension to be computed in.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr sum(int64_t dim = -1, bool lazy = true, TensorPtr outTensor = nullptr);

        /**
         * Computes the maximum in a given tensor dimension.
         * @param dim the dimension to be computed in.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr max(int64_t dim = -1, bool lazy = true, TensorPtr outTensor = nullptr);

        /**
         * Computes the minimum in a given tensor dimension.
         * @param dim the dimension to be computed in.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr min(int64_t dim = -1, bool lazy = true, TensorPtr outTensor = nullptr);

        /**
         * Permutes the shape of the tensor.
         * @param shapePerm the shape permutation consisting of indices from 0 to the size of the shape.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr perm(const std::vector<size_t> &shapePerm, bool lazy = true, TensorPtr outTensor = nullptr);

        /**
         * Transpose the tensor starting from the given dimension.
         * @param startDim the dimension to start transposing.
         * @param lazy whether the operation is executed lazily.
         * @param outTensor the output tensor.
         * @return the result tensor.
         */
        TensorPtr T(size_t startDim = 0, bool lazy = true, TensorPtr outTensor = nullptr);

        /**
         * Checks if the tensor is empty.
         * @return true if the tensor is empty and false otherwise.
         */
        bool isEmpty() const;

        /**
         * Forward propagation.
         */
        void forward();

        /**
         * Backward propagation.
         */
        void backward();
    };
}
