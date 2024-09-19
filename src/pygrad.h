//
// Created by Trung Luu on 9/14/24.
//

#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void init_vec_module(py::module_ &);
void init_shape_module(py::module_ &);
void init_tensor_module(py::module_ &);