//
// Created by Trung Luu on 9/14/24.
//

#include <sstream>
#include "pygrad.h"
#include "tensors/ops.h"

using namespace Toygrad::Tensor;

PYBIND11_MODULE(toygrad_cpu, m) {
    init_vec_module(m);
    init_shape_module(m);
    init_tensor_module(m);
    init_tensor_module(m);
}

void init_vec_module(py::module_ &m) {
    py::class_<Vec>(m, "Vec")
            .def(py::init<size_t>())
            .def(py::init<size_t, real>())
            .def(py::init<Vec &>())
            .def("__setitem__", [](const Vec &self, size_t index, real val) {
                self[index] = val;
            })
            .def("__getitem__", [](const Vec &self, size_t index) { return self[index]; });
}

void init_shape_module(py::module_ &m) {
    py::class_<Shape>(m, "Shape")
            .def_readonly("offset", &Shape::offset)
            .def_readonly("view", &Shape::view)
            .def_readonly("strides", &Shape::strides)
            .def(py::init<size_t, const std::vector<size_t> &, const std::vector<size_t> &>())
            .def(py::init<size_t, const std::vector<size_t> &>())
            .def(py::init<const std::vector<size_t> &>())
            .def("__len__", [](const Shape &self) { return self.getNumDims(); })
            .def("__getitem__", [](Shape &self, unsigned index) { return self[index]; });
}

void init_nn_module(py::module &m) {
}

void init_tensor_module(py::module_ &m) {
    py::class_<Tensor, std::shared_ptr<Tensor> >(m, "Tensor")
            .def("shape", &Tensor::getShape)
            .def("grad", &Tensor::getGrad)
            .def("__str__", [](const Tensor &self) {
                std::stringstream stream;
                stream << self << std::endl;
                return stream.str();
            })
            .def("is_contiguous", [](Tensor &self) { return self.isContiguous(); })
            .def("broadcast_to", [](Tensor &self, const Shape &shape) {
                return self.broadcastTo(shape);
            })
            .def("broadcast_to", [](Tensor &self, const std::vector<size_t> &view) {
                return self.broadcastTo(view);
            })
            .def("squeeze", [](Tensor &self, int64_t dim) {
                int64_t numDims = self.getShape().getNumDims();

                if (dim <= -numDims || dim >= numDims) {
                    throw py::index_error();
                }

                if (dim >= 0) {
                    return self.squeeze(dim);
                }

                return self.squeeze(numDims + dim);
            })
            .def("squeeze", [](Tensor &self) {
                return self.squeeze(-1);
            })
            .def("unsqueeze", [](Tensor &self, int64_t dim) {
                int64_t numDims = self.getShape().getNumDims();

                if (dim <= -numDims || dim >= numDims) {
                    throw py::index_error();
                }

                if (dim >= 0) {
                    return self.unsqueeze(dim);
                }

                return self.unsqueeze(numDims + dim);
            })
            .def("unsqueeze", [](Tensor &self) {
                return self.unsqueeze(-1);
            })
            .def_static("arange", [](const Shape &shape, real start, real step) {
                return Tensor::arange(shape, start, step);
            })
            .def_static("arange", [](const std::vector<size_t> &view, real start, real step) {
                return Tensor::arange(view, start, step);
            })
            .def_static("from_const", [](const Shape &shape, real c) {
                return Tensor::fromConst(shape, c);
            })
            .def_static("from_const", [](const std::vector<size_t> &view, real c) {
                return Tensor::fromConst(view, c);
            })
            .def_static("zeros", [](const Shape &shape) {
                return Tensor::zeros(shape);
            })
            .def_static("zeros", [](const std::vector<size_t> &view) {
                return Tensor::zeros(view);
            })
            .def_static("zeros_like", [](const Tensor &tensor) {
                return Tensor::zerosLike(tensor);
            })
            .def_static("ones", [](const Shape &shape) {
                return Tensor::ones(shape);
            })
            .def_static("ones", [](const std::vector<size_t> &view) {
                return Tensor::ones(view);
            })
            .def_static("ones_like", [](const Tensor &tensor) {
                return Tensor::onesLike(tensor);
            })
            .def_static("from_list", [](const Shape &shape, const std::vector<real> &data) {
                return Tensor::fromVec(shape, data);
            })
            .def_static("from_list", [](const std::vector<size_t> &view, const std::vector<real> &data) {
                return Tensor::fromVec(view, data);
            })
            .def_static("randn", [](const Shape &shape) {
                return Tensor::randn(shape);
            })
            .def_static("randn", [](const std::vector<size_t> &view) {
                return Tensor::randn(view);
            })
            .def_static("randint", [](const Shape &shape, int64_t min, int64_t max) {
                return Tensor::randint(shape, min, max);
            })
            .def_static("randint", [](const std::vector<size_t> &view, int64_t min, int64_t max) {
                return Tensor::randint(view, min, max);
            })
            .def("reshape", [](Tensor &self, const Shape &shape) {
                return self.reshape(shape);
            })
            .def("reshape", [](Tensor &self, const std::vector<size_t> &view) {
                return self.reshape(view);
            })
            .def("flatten", [](Tensor &self) {
                return self.flatten();
            })
            .def("perm", [](Tensor &self, const std::vector<size_t> &shapePerm) {
                return self.perm(shapePerm);
            })
            .def("T", [](Tensor &self, size_t startDim = 0) {
                return self.T(startDim);
            })
            .def("is_empty", [](const Tensor &self) {
                return self.isEmpty();
            })
            .def("__add__", [](Tensor &self, Tensor &rhs) {
                return self.add(rhs);
            })

            .def("__sub__", [](Tensor &self, Tensor &rhs) {
                return self.sub(rhs);
            })
            .def("__mul__", [](Tensor &self, Tensor &rhs) {
                return self.mul(rhs);
            })
            .def("__div__", [](Tensor &self, Tensor &rhs) {
                return self.div(rhs);
            })
            .def("matmul", [](Tensor &self, Tensor &rhs) {
                return self.matmul(rhs);
            })
            .def("__pow__", [](Tensor &self, real c) {
                return self.pow(c);
            })
            .def("log", [](Tensor &self) {
                return self.log();
            })
            .def("sin", [](Tensor &self) {
                return self.sin();
            })
            .def("cos", [](Tensor &self) {
                return self.cos();
            })
            .def("exp", [](Tensor &self) {
                return self.exp();
            })
            .def("sq", [](Tensor &self) {
                return self.sq();
            })
            .def("sqrt", [](Tensor &self) {
                return self.sqrt();
            })
            .def("neg", [](Tensor &self) {
                return self.neg();
            })
            .def("recip", [](Tensor &self, real c = 1) {
                return self.recip(c);
            })
            .def("__eq__", [](Tensor &self, Tensor &rhs) {
                return self.eq(rhs);
            })
            .def("__eq__", [](Tensor &self, real c) {
                return self.eq(c);
            })
            .def("__ne__", [](Tensor &self, Tensor &rhs) {
                return self.neq(rhs);
            })
            .def("__ne__", [](Tensor &self, real c) {
                return self.neq(c);
            })
            .def("__lt__", [](Tensor &self, Tensor &rhs) {
                return self.lt(rhs);
            })
            .def("__lt__", [](Tensor &self, real c) {
                return self.lt(c);
            })
            .def("__gt__", [](Tensor &self, Tensor &rhs) {
                return self.gt(rhs);
            })
            .def("__gt__", [](Tensor &self, real c) {
                return self.gt(c);
            })
            .def("__le__", [](Tensor &self, Tensor &rhs) {
                return self.leq(rhs);
            })
            .def("__le__", [](Tensor &self, real c) {
                return self.leq(c);
            })
            .def("__ge__", [](Tensor &self, Tensor &rhs) {
                return self.geq(rhs);
            })
            .def("__ge__", [](Tensor &self, real c) {
                return self.geq(c);
            })
            .def("__iadd__", [](Tensor &self, Tensor &rhs) {
                return self.addAssign(rhs);
            })
            .def("__iadd__", [](Tensor &self, real c) {
                return self.addAssign(c);
            })
            .def("__isub__", [](Tensor &self, Tensor &rhs) {
                return self.subAssign(rhs);
            })
            .def("__isub__", [](Tensor &self, real c) {
                return self.subAssign(c);
            })
            .def("__imul__", [](Tensor &self, Tensor &rhs) {
                return self.mulAssign(rhs);
            })
            .def("__imul__", [](Tensor &self, real c) {
                return self.mulAssign(c);
            })
            .def("__idiv__", [](Tensor &self, Tensor &rhs) {
                return self.divAssign(rhs);
            })
            .def("__idiv__", [](Tensor &self, real c) {
                return self.divAssign(c);
            })
            .def("relu", [](Tensor &self) {
                return self.relu();
            })
            .def("sigmoid", [](Tensor &self) {
                return self.sigmoid();
            })
            .def("softmax", [](Tensor &self, int64_t dim) {
                int64_t numDims = self.getShape().getNumDims();

                if (dim <= -numDims || dim >= numDims) {
                    throw py::index_error();
                }

                if (dim >= 0) {
                    return self.softmax(dim);
                }

                return self.softmax(numDims + dim);
            })
            .def("softmax", [](Tensor &self) {
                return self.softmax(-1);
            })
            .def("sum", [](Tensor &self, int64_t dim) {
                int64_t numDims = self.getShape().getNumDims();

                if (dim <= -numDims || dim >= numDims) {
                    throw py::index_error();
                }

                if (dim >= 0) {
                    return self.sum(dim);
                }

                return self.sum(numDims + dim);
            })
            .def("sum", [](Tensor &self) {
                return self.sum(-1);
            })
            .def("max", [](Tensor &self, int64_t dim) {
                int64_t numDims = self.getShape().getNumDims();

                if (dim <= -numDims || dim >= numDims) {
                    throw py::index_error();
                }

                if (dim >= 0) {
                    return self.max(dim);
                }

                return self.max(numDims + dim);
            })
            .def("max", [](Tensor &self) {
                return self.max(-1);
            })
            .def("min", [](Tensor &self, int64_t dim) {
                int64_t numDims = self.getShape().getNumDims();

                if (dim <= -numDims || dim >= numDims) {
                    throw py::index_error();
                }

                if (dim >= 0) {
                    return self.min(dim);
                }

                return self.min(numDims + dim);
            })
            .def("min", [](Tensor &self) {
                return self.min(-1);
            })
            .def("forward", [](Tensor &self) {
                self.forward();
            })
            .def("backward", [](Tensor &self) {
                self.backward();
            });
}
