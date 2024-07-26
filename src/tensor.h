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
        Op *op = nullptr;
        TensorPtr grad;
        std::vector<Tensor *> edges = std::vector<Tensor *>();

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
        friend struct ExpOp;
        friend struct RecipOp;
        friend struct NegOp;
        friend struct SqOp;
        friend struct EqOp;
        friend struct NeqOp;
        friend struct LessOp;
        friend struct GreaterOp;
        friend struct LeqOp;
        friend struct GeqOp;
        friend struct ReluOp;
        friend class TensorIndexer;

        Tensor();

        TensorPtr atHelper(const std::vector<size_t> &idx);

        void initGrad() {
            if (grad == nullptr) {
                grad = std::make_shared<Tensor>(shape);
            }
        }

    public:
        explicit Tensor(const Shape &shape);

        Tensor(const Shape &shape, const std::shared_ptr<Vec> &vec);

        Tensor(const Tensor &tensor);

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

        bool isContiguous() const;

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

        TensorPtr exp();

        TensorPtr recip(real c = 1);

        TensorPtr sq();

        TensorPtr neg();

        TensorPtr eq(const TensorPtr &rhs);

        TensorPtr eq(real c);

        bool operator==(Tensor &rhs);

        TensorPtr neq(const TensorPtr &rhs);

        TensorPtr neq(real c);

        bool operator!=(Tensor &rhs);

        TensorPtr lt(const TensorPtr &rhs);

        TensorPtr lt(real c);

        TensorPtr gt(const TensorPtr &rhs);

        TensorPtr gt(real c);

        TensorPtr leq(const TensorPtr &rhs);

        TensorPtr leq(real c);

        TensorPtr geq(const TensorPtr &rhs);

        TensorPtr geq(real c);

        Tensor &operator=(const Tensor &rhs);

        Tensor &operator=(real c);

        TensorPtr addAssign(const TensorPtr &rhs);

        TensorPtr addAssign(real c);

        TensorPtr subAssign(const TensorPtr &rhs);

        TensorPtr subAssign(real c);

        TensorPtr mulAssign(const TensorPtr &rhs);

        TensorPtr mulAssign(real c);

        TensorPtr divAssign(const TensorPtr &rhs);

        TensorPtr divAssign(real c);

        TensorPtr relu();

        TensorPtr reshape(const Shape &shape);

        TensorPtr sum();

        void backward();
    };
}
