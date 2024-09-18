#pragma once

#include "shape.h"
#include "vec.h"

namespace Toygrad::Tensor {
    class Tensor final : public std::enable_shared_from_this<Tensor> {
        Shape shape;
        std::shared_ptr<Vec> vec;
        static size_t idCounter;
        size_t id{};
        std::vector<Op *> ops = std::vector<Op *>();
        TensorPtr grad;
        std::vector<Tensor *> edges = std::vector<Tensor *>();

        friend class TensorGraph;
        friend struct Op;
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

        static TensorPtr initTensor(const Shape &shape, bool initStrides = true) {
            return std::make_shared<Tensor>(shape, initStrides);
        }

        TensorPtr index(const std::vector<size_t> &indices);

        TensorPtr index(const std::vector<Range> &ranges);

        void initVec() { vec = std::make_shared<Vec>(shape.getSize()); }

        void initVec(real c) { vec = std::make_shared<Vec>(shape.getSize(), c); }

        void initGrad() {
            if (grad == nullptr) {
                grad = initTensor(shape, false);
                grad->initVec();
            }
        }

        void initGrad(real c) {
            if (grad == nullptr) {
                grad = initTensor(shape, false);
                grad->initVec(c);
            }
        }

        bool isDimValid(int64_t dim) const { return dim >= -1 && dim < static_cast<int>(shape.getNumDims()); }

        TensorPtr perm(const Shape &target);

        TensorPtr alias(const Shape &target);

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
         * @return the result tensor.
         */
        TensorPtr broadcastTo(const Shape &target);

        /**
         * Broadcasts the tensor to a given shape.
         * @param view the view of the target shape to be broadcasted to.
         * @return the result tensor.
         */
        TensorPtr broadcastTo(const std::vector<size_t> &view) {
            return broadcastTo(Shape(view));
        }

        /**
         * Creates a shallow copy of the tensor using the same underlying memory.
         * @return a shallow copy of the tensor.
         */
        TensorPtr alias();

        /**
         * Creates a deep copy of the tensor using the same underlying memory.
         * @return a deep copy of the tensor.
         */
        TensorPtr copy();

        /**
         * Checks if the tensor is squeezable in a given dimension.
         * @param dim the dimension to squeeze the tensor.
         * @return true if the tensor is squeezable and false otherwise.
         */
        bool isSqueezable(int64_t dim = -1) const;

        /**
         * Squeezes the tensor in a given dimension.
         * @param dim the dimension to squeeze the tensor.
         * @return the result tensor.
         */
        TensorPtr squeeze(int64_t dim = -1);

        /**
         * Inserts(before) a dimension of size one at a position.
         * @param dim the dimension at which dimension of size one to be inserted. If dim is -1, it inserts the new
         * dimension at the end.
         * @return a new tensor with a dimension of size one inserted at the specified position.
         */
        TensorPtr unsqueeze(int64_t dim = -1);

        /**
         * Accesses tensor data at a given index.
         * @param idx the index to access the tensor at.
         * @return the result as a tensor.
         */
        TensorPtr at(size_t idx);

        /**
         * Accesses tensor data at given indices.
         * @param indices the indices to access the tensor at.
         * @return the result as a tensor.
         */
        TensorPtr at(const std::vector<size_t> &indices);

        /**
         * Accesses tensor data within given ranges.
         * @param ranges the ranges for accessing the tensor.
         * @return the result as a tensor.
         */
        TensorPtr at(const std::vector<Range> &ranges);

        /**
         * Creates a new tensor containing increasing integers.
         * @param shape the shape of the tensor.
         * @param start the starting integer.
         * @param step the step to increment each integer.
         * @return the newly created tensor.
         */
        static TensorPtr arange(const Shape &shape, real start, real step = 1.);

        /**
         * Creates a new tensor containing increasing integers.
         * @param view the view of the shape of the tensor.
         * @param start the starting integer.
         * @param step the step to increment each integer.
         * @return the newly created tensor.
         */
        static TensorPtr arange(const std::vector<size_t> &view, real start, real step = 1.) {
            return arange(Shape(view), start, step);
        }

        /**
         * Creates a new tensor containing random integers in the range[min, max].
         * @param shape the shape of the tensor.
         * @param min the lower-bound integer.
         * @param max the upper-bound integer.
         * @return the newly created tensor.
         */
        static TensorPtr randint(const Shape &shape, int64_t min, int64_t max);

        /**
         * Creates a new tensor containing random integers in the range[min, max].
         * @param view the view of the shape of the tensor.
         * @param min the lower-bound integer.
         * @param max the upper-bound integer.
         * @return the newly created tensor.
         */
        static TensorPtr randint(const std::vector<size_t> &view, int64_t min, int64_t max) {
            return randint(Shape(view), min, max);
        }

        /**
         * Creates a new tensor containing random values using normal distribution.
         * @param shape the shape of the tensor.
         * @return the newly created tensor.
         */
        static TensorPtr randn(const Shape &shape);

        /**
         * Creates a new tensor containing random values using normal distribution.
         * @param view the view of the shape of the tensor.
         * @return the newly created tensor.
         */
        static TensorPtr randn(const std::vector<size_t> &view) {
            return randn(Shape(view));
        }

        /**
         * Creates a new tensor containing the same constant value.
         * @param shape the shape of the tensor.
         * @param c the constant value.
         * @return the newly created tensor.
         */
        static TensorPtr fromConst(const Shape &shape, real c);

        /**
         * Creates a new tensor containing the same constant value.
         * @param view the view of the shape of the tensor.
         * @param c the constant value.
         * @return the newly created tensor.
         */
        static TensorPtr fromConst(const std::vector<size_t> &view, real c) {
            return fromConst(Shape(view), c);
        }

        /**
         * Creates a new tensor containing 0s.
         * @param shape the shape of the tensor.
         * @return the newly created tensor.
         */
        static TensorPtr zeros(const Shape &shape) {
            return fromConst(shape, 0.0);
        }

        /**
         * Creates a new tensor containing 0s.
         * @param view the view of the shape of the tensor.
         * @return the newly created tensor.
         */
        static TensorPtr zeros(const std::vector<size_t> &view) {
            return fromConst(view, 0.0);
        }

        /**
         * Creates a new tensor with 0s whose shape is the same as the given tensor.
         * @param tensor the tensor whose shape is the same as that of the new tensor.
         * @return the newly created tensor.
         */
        static TensorPtr zerosLike(const TensorPtr &tensor) {
            return zeros(tensor->shape);
        }

        /**
         * Creates a new tensor with 0s whose shape is the same as the given tensor.
         * @param tensor the tensor whose shape is the same as that of the new tensor.
         * @return the newly created tensor.
         */
        static TensorPtr zerosLike(const Tensor &tensor) {
            return zeros(tensor.shape);
        }

        /**
         * Creates a new tensor containing 1s.
         * @param shape the shape of the tensor.
         * @return the newly created tensor.
         */
        static TensorPtr ones(const Shape &shape) {
            return fromConst(shape, 1.0);
        }

        /**
         * Creates a new tensor containing 1s.
         * @param view the view of the shape of the tensor.
         * @return the newly created tensor.
         */
        static TensorPtr ones(const std::vector<size_t> &view) {
            return fromConst(view, 1.0);
        }

        /**
         * Creates a new tensor with 1s whose shape is the same as the given tensor.
         * @param tensor the tensor whose shape is the same as that of the new tensor.
         * @return the newly created tensor.
         */
        static TensorPtr onesLike(const TensorPtr &tensor) {
            return ones(tensor->shape);
        }

        /**
         * Creates a new tensor with 1s whose shape is the same as the given tensor.
         * @param tensor the tensor whose shape is the same as that of the new tensor.
         * @return the newly created tensor.
         */
        static TensorPtr onesLike(const Tensor &tensor) {
            return ones(tensor.shape);
        }

        /**
         * Constructs a tensor given a shape and an array.
         * @param shape the tensor shape.
         * @param data the array containing the tensor data.
         * @return a new tensor with the given shape and data.
         */
        static TensorPtr fromArr(const Shape &shape, const real *data);

        /**
         * Constructs a tensor given a shape and an array.
         * @param view the view of the tensor shape.
         * @param data the array containing the tensor data.
         * @return a new tensor with the given shape and data.
         */
        static TensorPtr fromArr(const std::vector<size_t> &view, const real *data) {
            return fromArr(Shape(view), data);
        }

        /**
         * Constructs a tensor given a shape and a vector.
         * @param shape the tensor shape.
         * @param data the vector containing the tensor data.
         * @return a new tensor with the given shape and data.
         */
        static TensorPtr fromVec(const Shape &shape, const std::vector<real> &data) {
            return fromArr(shape, data.data());
        }

        /**
         * Constructs a tensor given a shape and a vector.
         * @param view the view of the tensor shape.
         * @param data the vector containing the tensor data.
         * @return a new tensor with the given shape and data.
         */
        static TensorPtr fromVec(const std::vector<size_t> &view, const std::vector<real> &data) {
            return fromVec(Shape(view), data);
        }

        friend std::ostream &operator<<(std::ostream &stream, const Tensor &tensor);

        TensorPtr operator[](size_t idx);

        /**
         * Adds two tensors element-wise.
         * @param rhs the right tensor.
         * @return the result tensor.
         */
        TensorPtr add(const TensorPtr &rhs) { return add(*rhs); }

        /**
         * Adds two tensors element-wise.
         * @param rhs the right tensor.
         * @return the result tensor.
         */
        TensorPtr add(Tensor &rhs);

        /**
         * Adds a constant to each element in the tensor.
         * @param c the constant to be added.
         * @return the result tensor.
         */
        TensorPtr add(real c);

        /**
         * Subtracts two tensors element-wise.
         * @param rhs the right tensor.
         * @return the result tensor.
         */
        TensorPtr sub(const TensorPtr &rhs) { return sub(*rhs); }

        /**
         * Subtracts two tensors element-wise.
         * @param rhs the right tensor.
         * @return the result tensor.
         */
        TensorPtr sub(Tensor &rhs);

        /**
         * Subtracts a constant from each element in the tensor.
         * @param c the constant to be subtracted.
         * @return the result tensor.
         */
        TensorPtr sub(real c);

        /**
         * Multiplies two tensors element-wise.
         * @param rhs the right tensor.
         * @return the result tensor.
         */
        TensorPtr mul(const TensorPtr &rhs) { return mul(*rhs); }

        /**
         * Multiplies two tensors element-wise.
         * @param rhs the right tensor.
         * @return the result tensor.
         */
        TensorPtr mul(Tensor &rhs);

        /**
         * Multiplies each element by a constant in the tensor.
         * @param c the constant to be multiplied.
         * @return the result tensor.
         */
        TensorPtr mul(real c);

        /**
         * Divides two tensors element-wise.
         * @param rhs the right tensor.
         * @return the result tensor.
         */
        TensorPtr div(const TensorPtr &rhs) { return div(*rhs); }

        /**
         * Divides two tensors element-wise.
         * @param rhs the right tensor.
         * @return the result tensor.
         */
        TensorPtr div(Tensor &rhs);

        /**
         * Divides each element by a constant in the tensor.
         * @param c the constant to be divided.
         * @return the result tensor.
         */
        TensorPtr div(real c);

        /**
         * Raises each element in the tensor by a given power.
         * @param c the power to be raised.
         * @return the result tensor.
         */
        TensorPtr pow(real c);

        /**
         * Computes the natural logarithm of each element in the tensor.
         * @return the result tensor.
         */
        TensorPtr log();

        /**
         * Computes the sine of each element in the tensor.
         * @return the result tensor.
         */
        TensorPtr sin();

        /**
         * Computes the cosine of each element in the tensor.
         * @return the result tensor.
         */
        TensorPtr cos();

        /**
         * Computes the narutal exponent of each element in the tensor.
         * @return the result tensor.
         */
        TensorPtr exp();

        /**
         * Computes the reciprocal of each element in the tensor.
         * @return the result tensor.
         */
        TensorPtr recip(real c = 1);

        /**
         * Computes the square of each element in the tensor.
         * @return the result tensor.
         */
        TensorPtr sq();

        /**
         * Computes the square root of each element in the tensor.
         * @return the result tensor.
         */
        TensorPtr sqrt();

        /**
         * Computes the negation of each element in the tensor.
         * @return the result tensor.
         */
        TensorPtr neg();

        /**
         * Checks if two tensors are equal elementwise.
         * @param rhs the right tensor.
         * @return the result tensor.
         */
        TensorPtr eq(const TensorPtr &rhs) { return eq(*rhs); }

        /**
         * Checks if two tensors are equal elementwise.
         * @param rhs the right tensor.
         * @return the result tensor.
         */
        TensorPtr eq(Tensor &rhs);

        /**
         * Checks if each element in the tensor is equal to a constant.
         * @param c the constant to be compared.
         * @return the result tensor.
         */
        TensorPtr eq(real c);

        /**
         * Checks if tensors are equal elementwise.
         * @param rhs the right tensor.
         * @return true if all elements are equal and false otherwise.
         */
        bool operator==(const Tensor &rhs) const;

        /**
         * Checks if two tensors are not equal elementwise.
         * @param rhs the right tensor.
         * @return the result tensor.
         */
        TensorPtr neq(const TensorPtr &rhs) { return neq(*rhs); }

        /**
         * Checks if two tensors are not equal elementwise.
         * @param rhs the right tensor.
         * @return the result tensor.
         */
        TensorPtr neq(Tensor &rhs);

        /**
         * Checks if each element in the tensor is not equal to a constant.
         * @param c the constant to be compared.
         * @return the result tensor.
         */
        TensorPtr neq(real c);

        /**
         * Checks if tensors are not equal elementwise.
         * @param rhs the right tensor.
         * @return true if one pair of elements are not equal and false otherwise.
         */
        bool operator!=(const Tensor &rhs) const;

        /**
         * Checks if the left tensor is less than the right tensor elementwise.
         * @param rhs the right tensor.
         * @return the result tensor.
         */
        TensorPtr lt(const TensorPtr &rhs) { return lt(*rhs); }

        /**
         * Checks if the left tensor is less than the right tensor elementwise.
         * @param rhs the right tensor.
         * @return the result tensor.
         */
        TensorPtr lt(Tensor &rhs);

        /**
         * Checks if each element of the tensor is less than a constant.
         * @param c the constant to be compared.
         * @return the result tensor.
         */
        TensorPtr lt(real c);

        /**
         * Checks if the left tensor is greater than the right tensor elementwise.
         * @param rhs the right tensor.
         * @return the result tensor.
         */
        TensorPtr gt(const TensorPtr &rhs) { return gt(*rhs); }

        /**
         * Checks if the left tensor is greater than the right tensor elementwise.
         * @param rhs the right tensor.
         * @return the result tensor.
         */
        TensorPtr gt(Tensor &rhs);

        /**
         * Checks if each element of the tensor is greater than a constant.
         * @param c the constant to be compared.
         * @return the result tensor.
         */
        TensorPtr gt(real c);

        /**
         * Checks if the left tensor is less than or equal to the right tensor elementwise.
         * @param rhs the right tensor.
         * @return the result tensor.
         */
        TensorPtr leq(const TensorPtr &rhs) { return leq(*rhs); }

        /**
         * Checks if the left tensor is less than or equal to the right tensor elementwise.
         * @param rhs the right tensor.
         * @return the result tensor.
         */
        TensorPtr leq(Tensor &rhs);

        /**
         * Checks if each element of the tensor is less than or equal to a constant.
         * @param c the constant to be compared.
         * @return the result tensor.
         */
        TensorPtr leq(real c);

        /**
         * Checks if the left tensor is greater than or equal to the right tensor elementwise.
         * @param rhs the right tensor.
         * @return the result tensor.
         */
        TensorPtr geq(const TensorPtr &rhs) { return geq(*rhs); }

        /**
         * Checks if the left tensor is greater than or equal to the right tensor elementwise.
         * @param rhs the right tensor.
         * @return the result tensor.
         */
        TensorPtr geq(Tensor &rhs);

        /**
         * Checks if each element of the tensor is greater than or equal to a constant.
         * @param c the constant to be compared.
         * @return the result tensor.
         */
        TensorPtr geq(real c);

        Tensor &operator=(const Tensor &rhs) = delete;

        /**
         * Increments each element in the current tensor by the corresponding element in the right tensor.
         * @param rhs the right tensor.
         * @return the result tensor, the same as the current tensor.
         */
        TensorPtr addAssign(const TensorPtr &rhs) { return addAssign(*rhs); }

        /**
         * Increments each element in the current tensor by the corresponding element in the right tensor.
         * @param rhs the right tensor.
         * @return the result tensor, the same as the current tensor.
         */
        TensorPtr addAssign(Tensor &rhs);

        /**
         * Increments each element in the current tensor by a constant.
         * @param c the constant to be incremented by.
         * @return the result tensor, the same as the current tensor.
         */
        TensorPtr addAssign(real c);

        /**
         * Decrements each element in the current tensor by the corresponding element in the right tensor.
         * @param rhs the right tensor.
         * @return the result tensor, the same as the current tensor.
         */
        TensorPtr subAssign(const TensorPtr &rhs) { return subAssign(*rhs); }

        /**
         * Decrements each element in the current tensor by the corresponding element in the right tensor.
         * @param rhs the right tensor.
         * @return the result tensor, the same as the current tensor.
         */
        TensorPtr subAssign(Tensor &rhs);

        /**
         * Decrements each element in the current tensor by a constant.
         * @param c the constant to be decremented by.
         * @return the result tensor, the same as the current tensor.
         */
        TensorPtr subAssign(real c);

        /**
         * Multiplies in-place each element in the current tensor by the corresponding element in the right tensor.
         * @param rhs the right tensor.
         * @return the result tensor, the same as the current tensor.
         */
        TensorPtr mulAssign(const TensorPtr &rhs) { return mulAssign(*rhs); }

        /**
         * Multiplies in-place each element in the current tensor by the corresponding element in the right tensor.
         * @param rhs the right tensor.
         * @return the result tensor, the same as the current tensor.
         */
        TensorPtr mulAssign(Tensor &rhs);

        /**
         * Multiplies in-place each element in the current tensor by a constant.
         * @param c the constant to be multiplied by.
         * @return the result tensor, the same as the current tensor.
         */
        TensorPtr mulAssign(real c);

        /**
         * Divides in-place each element in the current tensor by the corresponding element in the right tensor.
         * @param rhs the right tensor.
         * @return the result tensor, the same as the current tensor.
         */
        TensorPtr divAssign(const TensorPtr &rhs) { return divAssign(*rhs); }

        /**
         * Divides in-place each element in the current tensor by the corresponding element in the right tensor.
         * @param rhs the right tensor.
         * @return the result tensor, the same as the current tensor.
         */
        TensorPtr divAssign(Tensor &rhs);

        /**
         * Divides in-place each element in the current tensor by a constant.
         * @param c the constant to be divided by.
         * @return the result tensor, the same as the current tensor.
         */
        TensorPtr divAssign(real c);

        /**
         * Computes Rectified Linear Unit(ReLU) elementwise in the tensor.
         * @return the result tensor.
         */
        TensorPtr relu();

        /**
         * Computes Sigmoid elementwise in the tensor.
         * @return the result tensor.
         */
        TensorPtr sigmoid();

        /**
         * Computes Softmax in a given tensor dimension.
         * @param dim the dimension to be computed in.
         * @return the result tensor.
         */
        TensorPtr softmax(int64_t dim = -1);

        /**
         * Matrix multiplies two tensors in the last two dimensions.
         * @param rhs the right tensor.
         * @return the result tensor.
         */
        TensorPtr matmul(const TensorPtr &rhs) { return matmul(*rhs); }

        /**
         * Matrix multiplies two tensors in the last two dimensions.
         * @param rhs the right tensor.
         * @return the result tensor.
         */
        TensorPtr matmul(Tensor &rhs);

        /**
         * Reshapes the tensor to a given shape.
         * @param target the target shape to be reshaped to.
         * @return the result tensor.
         */
        TensorPtr reshape(const Shape &target);

        /**
         * Reshapes the tensor to a given shape.
         * @param view the view of the target shape to be reshaped to.
         * @return the result tensor.
         */
        TensorPtr reshape(const std::vector<size_t> &view) {
            return reshape(Shape(view));
        }

        /**
         * Flattens the tensor.
         * @return the result tensor.
         */
        TensorPtr flatten() { return reshape({shape.getSize()}); }

        /**
         * Computes the summation in a given tensor dimension.
         * @param dim the dimension to be computed in.
         * @return the result tensor.
         */
        TensorPtr sum(int64_t dim = -1);

        /**
         * Computes the maximum in a given tensor dimension.
         * @param dim the dimension to be computed in.
         * @return the result tensor.
         */
        TensorPtr max(int64_t dim = -1);

        /**
         * Computes the minimum in a given tensor dimension.
         * @param dim the dimension to be computed in.
         * @return the result tensor.
         */
        TensorPtr min(int64_t dim = -1);

        /**
         * Permutes the shape of the tensor.
         * @param shapePerm the shape permutation consisting of indices from 0 to the size of the shape.
         * @return the result tensor.
         */
        TensorPtr perm(const std::vector<size_t> &shapePerm);

        /**
         * Transpose the tensor starting from the given dimension.
         * @param startDim the dimension to start transposing.
         * @return the result tensor.
         */
        TensorPtr T(size_t startDim = 0);

        /**
         * Checks if the tensor is empty.
         * @return true if the tensor is empty and false otherwise.
         */
        bool isEmpty() const;

        /**
         * Forward propagation.
         */
        void forward() const;

        /**
         * Backward propagation.
         */
        void backward() const;
    };
}
