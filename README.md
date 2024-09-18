# Toygrad (CPU)

## Introduction

A simple and toy framework similar to PyTorch written in C++ 20. Currently, it only works on CPU and is not optimized
for performance. There are still bugs internally so this project is not intended to be used in production.

## :raised_hands: Acknowledgements

* https://martinlwx.github.io/en/how-to-reprensent-a-tensor-or-ndarray/#fn:2
* http://blog.ezyang.com/2019/05/pytorch-internals/
* https://dlsyscourse.org/lectures/

## :white_check_mark: Requirements

* CMake 3.28 or higher
* C++ 20
* Googletest library(optional)
* Git
* Pybind11 2.13.5
* Python 3.12(see Python versions supported by PyBind11)

## :rocket: Quick start

* Insert path to `Python.h` and the include path of Pybind11 in CMAKE. It is best that they are from the same Python
  environment and using the same interpreter.
* Build using the following command in the project folder:

```
cmake -S . -B build
cmake --build build
```

## :rocket: Features

### Implemented

* Element-wise add, sub, mul, div
* Negation
* Reciprocal
* Exponent
* Square
* Square root
* Log
* Pow
* Sin, cos
* Tensor indexing
* Randint
* Randn
* Arange
* Tensor with the same constant
* Tensor from an array or vector with the given shape
* ==, !=, <, >, <=, >=
* =, +=, -=, *=, /=
* Shape permutation
* Transpose
* Max, min
* Broadcasting
* Squeeze, unsqueeze
* Sum
* Relu
* Sigmoid
* Softmax
* Matmul: executes similarly to PyTorch where tensor multiplication is only applied in the last two
  dimensions
* Backprop
* Lazy execution: waits until the computational graph is forwarded to compute tensor values

### In progress

* Python support
* Stack: not implemented
* Cat: not implemented
* Flatten: not tested
* Reshape: currently shares memory when the tensor is contiguous but allocates new memory when the tensor is
  non-contiguous

## :computer: Code

### C++

```
// Arange operation
std::vector<size_t> v1 = {2, 3, 4};
Shape s1(v1);
auto t1 = Tensor::arange(s1, 0); // 0 means starting at 0
std::cout << *t1 << std::endl;

Output:
[[[0, 1, 2, 3], 
[4, 5, 6, 7], 
[8, 9, 10, 11]], 
[[12, 13, 14, 15], 
[16, 17, 18, 19], 
[20, 21, 22, 23]]]

// Summation by dimension
auto t2 = t1->sum(1);
std::cout << *t2 << std::endl;

Output:
[[12, 15, 18, 21], 
[48, 51, 54, 57]]

// Shape permutation, in this case, same as transpose
auto t3 = t1->perm({2, 1, 0});
std::cout << *t3 << std::endl;

Output:
[[[0, 12], 
[4, 16], 
[8, 20]], 
[[1, 13], 
[5, 17], 
[9, 21]], 
[[2, 14], 
[6, 18], 
[10, 22]], 
[[3, 15], 
[7, 19], 
[11, 23]]]

// Tensor indexing
Range r1 = {0, 2, 1};
Range r2 = {1, 3, 2};
Range r3 = {0, 4, 2};
std::vector<Range> q2 = {r1, r2, r3};
auto t4 = t1->at(q2);
std::cout << *t4 << std::endl;

Output:
[[[4, 6]], 
[[16, 18]]]

// Tensor softmax
auto t5 = t4->softmax(0);
std::cout << *t5 << std::endl;

Output:
[[[6.14417e-06, 6.14417e-06]], 
[[0.999994, 0.999994]]]

// Matrix multiplication
std::vector<size_t> v6 = {2, 3};
Shape s6(v6);
auto t6 = Tensor::arange(s6, 0);
std::vector<size_t> v7 = {3, 4};
Shape s7(v7);
auto t7 = Tensor::arange(s7, 0);
auto t8 = t6->matmul(t7);
std::cout << *t8 << std::endl;

Output:
[[20, 23, 26, 29], 
[56, 68, 80, 92]]
```

### Python

```
from toygrad_cpu import Tensor, Shape, TensorGraph

t1 = Tensor.randn([2, 3, 4])
t2 = Tensor.randn([2, 3, 4])
t3 = t1 + t2
graph = TensorGraph.from_tensor(t3)
graph.forward()
print(t1)
print(t2)
print(t3)

Output:
[[[-2.14481, -0.327103, 0.827069, -0.139807], 
[0.428268, -0.0396302, 0.54151, -2.92291], 
[-0.849546, 1.12644, -1.30544, 0.0385326]], 
[[-0.517463, 0.4214, 0.213025, -0.475141], 
[-0.739245, -0.784967, 0.159765, 0.600664], 
[0.687566, -0.435637, 1.24142, -0.247375]]]

[[[-2.72279, -0.745629, 1.01239, -1.05384], 
[1.72556, -0.661698, 0.378033, 0.0188341], 
[-0.486968, 0.298707, -0.205761, -0.185135]], 
[[-1.25837, 1.42416, -0.448031, 1.50035], 
[-0.312271, 1.99179, -2.39648, 1.66187], 
[1.19221, 0.0519404, -0.44068, 0.512235]]]

[[[-4.8676, -1.07273, 1.83946, -1.19365], 
[2.15382, -0.701329, 0.919543, -2.90407], 
[-1.33651, 1.42515, -1.5112, -0.146602]], 
[[-1.77583, 1.84556, -0.235005, 1.02521], 
[-1.05152, 1.20683, -2.23671, 2.26253], 
[1.87978, -0.383697, 0.800736, 0.264861]]]

import faulthandler
from toygrad_cpu import Tensor, Shape, TensorGraph

faulthandler.enable()
t1 = Tensor.randn([2, 3, 4])
t2 = Tensor.randn([2, 3, 4])
t3 = t1.max(1)
t4 = t3.sum()
graph = TensorGraph.from_tensor(t4)
graph.forward()
graph.backward()
print(t1)
print(t3)
print(t1.grad())

[[[0.360319, 0.492564, -1.4646, -1.43054], 
[-0.436966, -0.955127, -0.588588, -0.810968], 
[-0.736762, 0.309234, 0.331526, 0.214696]], 
[[0.826028, -0.696271, 0.74823, 0.155481], 
[0.062263, -0.403605, -1.47178, 1.82771], 
[1.43528, 0.116985, 1.13029, -0.0663171]]]

[[0.360319, 0.492564, 0.331526, 0.214696], 
[1.43528, 0.116985, 1.13029, 1.82771]]

[[[1, 1, 0, 0], 
[0, 0, 0, 0], 
[0, 0, 1, 1]], 
[[0, 0, 0, 0], 
[0, 0, 0, 1], 
[1, 1, 1, 0]]]
```