#pragma once

#include "shape.h"
#include "vec.h"
#include "str_assert.h"

namespace Toygrad::Tensor {
    class Tensor final : public std::enable_shared_from_this<Tensor> {
        Shape shape;
        std::shared_ptr<Vec> vec;
        static size_t idCounter;
        size_t id;
        std::vector<Op *> ops = std::vector<Op *>();
        TensorPtr grad;
        std::vector<Tensor *> edges = std::vector<Tensor *>();

        friend class CGraph;
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

        Tensor();

        TensorPtr index(const std::vector<size_t> &indices);

        TensorPtr index(const std::vector<Range> &ranges);

        void initVec() {
            vec = std::make_shared<Vec>(shape.getSize());
        }

        void initVec(real c) {
            vec = std::make_shared<Vec>(shape.getSize(), c);
        }

        void initGrad() {
            if (grad == nullptr) {
                grad = std::make_shared<Tensor>(shape);
                grad->initVec();
            }
        }

        void initGrad(real c) {
            if (grad == nullptr) {
                grad = std::make_shared<Tensor>(shape);
                grad->initVec(c);
            }
        }

        bool isDimValid(int dim) const {
            return dim >= -1 && dim < static_cast<int>(shape.getNumDims());
        }

        TensorPtr perm(const Shape &target);

        TensorPtr alias(const Shape &target);

    public:
        explicit Tensor(const Shape &shape, bool initStrides = true);

        Tensor(const Tensor &tensor) = delete;

        ~Tensor();

        size_t getId() const {
            return id;
        }

        /**
         * Gets the shape of the tensor.
         * @return the shape of the current tensor.
         */
        const Shape &getShape() const {
            return shape;
        }

        /**
         * Gets the gradient of the current tensor.
         * @return the gradient tensor of the current tensor.
         */
        TensorPtr getGrad() const {
            return grad;
        }

        /**
         * Gets a pointer to the underlying memory.
         * @return a pointer to the underlying memory.
         */
        std::shared_ptr<Vec> getVec() const {
            return vec;
        }

        /**
         * Checks if the tensor's memory is contiguous.
         * @return true if the underlying memory is contiguous and false otherwise.
         */
        bool isContiguous() const;

        bool isBroadcastableTo(const Shape &target) const;

        TensorPtr broadcastTo(const Shape &target);

        TensorPtr alias();

        TensorPtr copy();

        bool isSqueezable(int dim = -1) const;

        TensorPtr squeeze(int dim = -1);

        /**
         * Inserts(before) a dimension of size one at a position.
         * @param dim the dimension at which dimension of size one to be inserted. If dim is -1, it inserts the new
         * dimension at the end.
         * @return a new tensor with a dimension of size one inserted at the specified position.
         */
        TensorPtr unsqueeze(int dim = -1);

        TensorPtr at(size_t idx);

        TensorPtr at(const std::vector<size_t> &indices);

        TensorPtr at(const std::vector<Range> &ranges);

        static TensorPtr arange(const Shape &shape, real start, real step = 1.);

        static TensorPtr randint(const Shape &shape, int min, int max);

        static TensorPtr randn(const Shape &shape);

        static TensorPtr fromConst(const Shape &shape, real c);

        static TensorPtr fromArr(const Shape &shape, const real *data);

        friend std::ostream &operator<<(std::ostream &stream, Tensor &tensor);

        TensorPtr operator[](size_t idx);

        /**
         * Adds two tensors element-wise.
         * @param rhs the right tensor.
         * @return the result tensor.
         */
        TensorPtr add(const TensorPtr &rhs);

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
        TensorPtr sub(const TensorPtr &rhs);

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
        TensorPtr mul(const TensorPtr &rhs);

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
        TensorPtr div(const TensorPtr &rhs);

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
        TensorPtr eq(const TensorPtr &rhs);

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
        TensorPtr neq(const TensorPtr &rhs);

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
        TensorPtr lt(const TensorPtr &rhs);

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
        TensorPtr gt(const TensorPtr &rhs);

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
        TensorPtr leq(const TensorPtr &rhs);

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
        TensorPtr geq(const TensorPtr &rhs);

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
        TensorPtr addAssign(const TensorPtr &rhs);

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
        TensorPtr subAssign(const TensorPtr &rhs);

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
        TensorPtr mulAssign(const TensorPtr &rhs);

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
        TensorPtr divAssign(const TensorPtr &rhs);

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
        TensorPtr softmax(int dim = -1);

        TensorPtr reshape(const Shape &target);

        /**
         * Computes the summation in a given tensor dimension.
         * @param dim the dimension to be computed in.
         * @return the result tensor.
         */
        TensorPtr sum(int dim = -1);

        /**
         * Computes the maximum in a given tensor dimension.
         * @param dim the dimension to be computed in.
         * @return the result tensor.
         */
        TensorPtr max(int dim = -1);

        /**
         * Computes the minimum in a given tensor dimension.
         * @param dim the dimension to be computed in.
         * @return the result tensor.
         */
        TensorPtr min(int dim = -1);

        /**
         * Permutes the shape of the tensor.
         * @param shapePerm the shape permutation consisting of indices from 0 to the size of the shape.
         * @return the result tensor.
         */
        TensorPtr perm(const std::vector<size_t> &shapePerm);

        /**
         * Transpose the tensor.
         * @return the result tensor.
         */
        TensorPtr T();

        /**
         * Checks if the tensor is empty.
         * @return true if the tensor is empty and false otherwise.
         */
        bool isEmpty();

        /**
         * Backpropagates and initializes the tensor's gradient.
         */
        void backward();
    };
}
