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

        TensorPtr index(const std::vector<size_t> &idx);

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

        // Shape getNewShape();

    public:
        explicit Tensor(const Shape &shape, bool newTensor = true);

        Tensor(const Tensor &tensor) = delete;

        ~Tensor();

        size_t getId() const {
            return id;
        }

        const Shape &getShape() const {
            return shape;
        }

        TensorPtr getGrad() const {
            return grad;
        }

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

        TensorPtr at(const std::vector<size_t> &idx);

        TensorPtr at(const std::vector<Range> &ranges);

        static TensorPtr arange(const Shape &shape, real start, real step = 1.);

        static TensorPtr randint(const Shape &shape, int min, int max);

        static TensorPtr randn(const Shape &shape);

        static TensorPtr fromConst(const Shape &shape, real c);

        static TensorPtr fromArr(const Shape &shape, const real *data);

        friend std::ostream &operator<<(std::ostream &stream, Tensor &tensor);

        TensorPtr operator[](size_t idx);

        TensorPtr add(const TensorPtr &rhs);

        TensorPtr add(real c);

        TensorPtr sub(const TensorPtr &rhs);

        TensorPtr sub(real c);

        TensorPtr mul(const TensorPtr &rhs);

        TensorPtr mul(real c);

        TensorPtr div(const TensorPtr &rhs);

        TensorPtr div(real c);

        TensorPtr pow(real c);

        TensorPtr log();

        TensorPtr sin();

        TensorPtr cos();

        TensorPtr exp();

        TensorPtr recip(real c = 1);

        TensorPtr sq();

        TensorPtr sqrt();

        TensorPtr neg();

        TensorPtr eq(const TensorPtr &rhs);

        TensorPtr eq(real c);

        bool operator==(const Tensor &rhs) const;

        TensorPtr neq(const TensorPtr &rhs);

        TensorPtr neq(real c);

        bool operator!=(const Tensor &rhs) const;

        TensorPtr lt(const TensorPtr &rhs);

        TensorPtr lt(real c);

        TensorPtr gt(const TensorPtr &rhs);

        TensorPtr gt(real c);

        TensorPtr leq(const TensorPtr &rhs);

        TensorPtr leq(real c);

        TensorPtr geq(const TensorPtr &rhs);

        TensorPtr geq(real c);

        Tensor &operator=(const Tensor &rhs) = delete;

        TensorPtr addAssign(const TensorPtr &rhs);

        TensorPtr addAssign(real c);

        TensorPtr subAssign(const TensorPtr &rhs);

        TensorPtr subAssign(real c);

        TensorPtr mulAssign(const TensorPtr &rhs);

        TensorPtr mulAssign(real c);

        TensorPtr divAssign(const TensorPtr &rhs);

        TensorPtr divAssign(real c);

        TensorPtr relu();

        TensorPtr sigmoid();

        TensorPtr softmax(int dim = -1);

        TensorPtr reshape(const Shape &target);

        TensorPtr sum(int dim = -1);

        TensorPtr max(int dim = -1);

        TensorPtr min(int dim = -1);

        TensorPtr perm(const std::vector<size_t> &shapePerm);

        TensorPtr T();

        bool isEmpty();

        void backward();
    };
}
