# Toygrad (CPU)

## Introduction

A simple and toy framework similar to PyTorch written in C++ 20. Currently, it only works on CPU and is not optimized
for performance. There are still bugs internally so this project is not intended to be used in production.

## Acknowledgements

* https://martinlwx.github.io/en/how-to-reprensent-a-tensor-or-ndarray/#fn:2
* http://blog.ezyang.com/2019/05/pytorch-internals/
* https://dlsyscourse.org/lectures/

## Requirements

* CMake 3.28 or higher
* C++ 20
* Googletest library(optional)
* Git

## Quick start

For the CMAKE file, we can ignore the tests directory since it is used for testing only. Hence, we can comment out the
following line in the main CMAKE file:

```
cmake_minimum_required(VERSION 3.28)
project(toygrad_cpu)

set(CMAKE_CXX_STANDARD 20)

include_directories(src)
add_subdirectory(src)
// add_subdirectory(tests)
add_executable(toygrad_cpu main.cpp)
target_link_libraries(toygrad_cpu toygrad_cpu_lib)
```

and remove the `tests` folder.

You can build using the following command in the project folder:

```
cmake -S . -B build
cmake --build build
```

## Features

- [x] tensor-to-tensor element-wise add, sub, mul, div
- [x] tensor-to-constant add, sub, mul, div
- [x] negation
- [x] reciprocal
- [x] exponent
- [x] square
- [x] square root
- [x] log
- [x] pow
- [x] sin, cos
- [x] tensor indexing
- [x] randint
- [x] randn
- [x] arange
- [ ] tensor with constant: currently non-optimal
- [x] tensor from an array with the given shape
- [x] ==, !=, <, >, <=, >=
- [x] =, +=, -=, *=, /=
- [x] shape permutation
- [x] transpose
- [x] max, min
- [x] broadcasting
- [ ] squeeze, unsqueeze
- [x] sum
- [x] relu
- [x] sigmoid
- [ ] reshape: currently shares memory when the tensor is contiguous but allocates new memory when the tensor is
  non-contiguous
- [ ] stack
- [ ] cat
- [x] softmax
- [ ] matmul

## Code

```
// Arange operation
std::vector<size_t> v = {2, 3, 4};
Shape s1(v);
auto t1 = Tensor::arange(s1, 0); // 0 means starting at 0

Output:
[[[0, 1, 2, 3], 
[4, 5, 6, 7], 
[8, 9, 10, 11]], 
[[12, 13, 14, 15], 
[16, 17, 18, 19], 
[20, 21, 22, 23]]]

// Summation by dimension
auto t2 = t1->sum(1);

Output:
[[12, 15, 18, 21], 
[48, 51, 54, 57]]

// Shape permutation, in this case, same as transpose
auto t3 = t1->perm({2, 1, 0});

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

Output:
[[[4, 6]], 
[[16, 18]]]
```

## Backpropagation

Backpropagation can be done by calling `tensor.backward()`. This is done by iterating the computational graph in
reversed topological order.