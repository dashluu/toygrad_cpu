//
// Created by Trung Luu on 7/16/24.
//

#include "graph.h"
#include "gtest/gtest.h"

using namespace Toygrad::Tensor;

class TensorTestFixture : public testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {
        Graph::cleanUp();
    }
};

TEST(TensorTestFixture, indexTensor1) {
    std::vector<size_t> v = {2, 3, 4};
    Range r1 = {0, 2, 1};
    Range r2 = {4, 3, 2};
    Range r3 = {0, 4, 2};
    std::vector<size_t> q1 = {1, 1};
    std::vector<Range> q2 = {r1, r2, r3};
    Shape s(v);
    auto t1 = Tensor::arange(s, 0);
    auto t2 = t1->at(q1);
    auto t3 = t1->at(q2);
    std::cout << *t1 << std::endl;
    std::cout << *t2 << std::endl;
    std::cout << *t3 << std::endl;
}
