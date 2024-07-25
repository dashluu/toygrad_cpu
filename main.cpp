#include <iostream>

#include "tensor.h"
#include "graph.h"

using namespace Toygrad::Tensor;

int main() {
    std::vector<size_t> v = {2, 3, 4};
    Range r1 = {0, 2, 1};
    Range r2 = {4, 3, 2};
    Range r3 = {0, 4, 2};
    std::vector<size_t> q1 = {1, 1};
    std::vector<Range> q2 = {r1, r2, r3};
    Shape s(v);
    Tensor &t1 = Tensor::arange(s, 0);
    Tensor &t2 = Tensor::randint(s, 1, 10);
    Tensor &t3 = t1.at(q1);
    Tensor &t4 = t1.at(q2);
    std::cout << t1 << std::endl;
    // std::cout << t2 << std::endl;
    std::cout << t3 << std::endl;
    std::cout << t4 << std::endl;

    // std::vector<size_t> v = {2, 3, 4, 5};
    // Range r1 = {1, 2, 2};
    // Range r2 = {1, 3, 2};
    // Range r3 = {1, 4, 2};
    // Range r4 = {1, 5, 1};
    // Range r5 = {0, 1, 1};
    // Range r6 = {0, 1, 1};
    // Range r7 = {0, 2, 1};
    // Range r8 = {1, 4, 2};
    // std::vector<size_t> q1 = {0, 0, 1};
    // std::vector q2 = {r1, r2, r3, r4};
    // std::vector q3 = {r5, r6, r7, r8};
    // Shape s(v);
    // Tensor &t1 = Tensor::arange(s, 0);
    // // Tensor &t2 = Tensor::randint(s, 1, 10);
    // Tensor &t4 = t1.at(q2);
    // Tensor &t3 = t4.at(q3);
    // std::cout << t1 << std::endl << std::endl;
    // // std::cout << t2 << std::endl;
    // std::cout << t4 << std::endl << std::endl;
    // std::cout << t3 << std::endl;
    Graph::cleanUp();
    return 0;
}
