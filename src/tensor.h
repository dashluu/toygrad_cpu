#pragma once

#include "shape.h"
#include "vec.h"
#include "str_assert.h"

namespace Toygrad::Tensor {
    class Tensor final {
        Shape shape;
        Vec *vec = nullptr;
        static size_t idCounter;
        size_t id;
        Op *op = nullptr;

        friend struct Op;
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
        friend struct EqOp;
        friend struct NeqOp;
        friend struct LessOp;
        friend struct GreaterOp;
        friend struct LeqOp;
        friend struct GeqOp;
        friend struct ReluOp;
        friend class TensorIndexer;

        Tensor();

        explicit Tensor(const Shape &shape);

        Tensor(const Shape &shape, Vec *vec);

        Tensor *atHelper(const std::vector<size_t> &idx);

    public:
        Tensor(const Tensor &tensor);

        ~Tensor();

        size_t getId() const {
            return id;
        }

        const Shape &getShape() const {
            return shape;
        }

        inline Tensor &getGrad() const;

        const Vec *getVec() const {
            return vec;
        }

        bool isContiguous() const;

        Tensor &at(const std::vector<size_t> &idx);

        Tensor &at(const std::vector<Range> &ranges);

        static Tensor &arange(const Shape &shape, real start, real step = 1.);

        static Tensor &randint(const Shape &shape, int min, int max);

        static Tensor &randn(const Shape &shape);

        static Tensor &fromConst(const Shape &shape, real c);

        static Tensor &fromArr(const Shape &shape, const real *data);

        friend std::ostream &operator<<(std::ostream &stream, Tensor &tensor);

        Tensor &operator[](size_t idx);

        Tensor &operator+(Tensor &rhs);

        Tensor &operator+(real c);

        Tensor &operator-(Tensor &rhs);

        Tensor &operator-(real c);

        Tensor &operator*(Tensor &rhs);

        Tensor &operator*(real c);

        Tensor &operator/(Tensor &rhs);

        Tensor &operator/(real c);

        Tensor &exp();

        Tensor &eq(Tensor &rhs);

        Tensor &eq(real c);

        bool operator==(Tensor &rhs);

        Tensor &neq(Tensor &rhs);

        Tensor &neq(real c);

        bool operator!=(Tensor &rhs);

        Tensor &operator<(Tensor &rhs);

        Tensor &operator<(real c);

        Tensor &operator>(Tensor &rhs);

        Tensor &operator>(real c);

        Tensor &operator<=(Tensor &rhs);

        Tensor &operator<=(real c);

        Tensor &operator>=(Tensor &rhs);

        Tensor &operator>=(real c);

        Tensor &operator=(const Tensor &rhs);

        Tensor &operator=(real c);

        Tensor &operator+=(Tensor &rhs);

        Tensor &operator+=(real c);

        Tensor &operator-=(Tensor &rhs);

        Tensor &operator-=(real c);

        Tensor &operator*=(Tensor &rhs);

        Tensor &operator*=(real c);

        Tensor &operator/=(Tensor &rhs);

        Tensor &operator/=(real c);

        friend Tensor &operator+(real c, Tensor &rhs);

        friend Tensor &operator-(real c, Tensor &rhs);

        friend Tensor &operator*(real c, Tensor &rhs);

        friend Tensor &operator/(real c, Tensor &rhs);

        Tensor &relu();

        Tensor &reshape(const Shape &shape);

        Tensor &sum();

        void backward();
    };
}
