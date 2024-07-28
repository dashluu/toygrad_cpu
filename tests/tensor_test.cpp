//
// Created by Trung Luu on 7/16/24.
//

#include "gtest/gtest.h"
#include "tensor.h"

using namespace Toygrad::Tensor;

class TensorTestFixture : public testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {
    }
};

void testTemplate(Tensor &actual, Tensor &expected) {
    std::cout << "Actual:" << std::endl << actual << std::endl;
    std::cout << "Expected:" << std::endl << expected << std::endl;
    ASSERT_EQ(actual, expected);
}

TEST(TensorTestFixture, indexTensor1) {
    std::cout << std::endl << "Indexing tensor 1:" << std::endl;
    std::vector<size_t> v = {2, 3, 4};
    Range r1 = {0, 2, 1};
    Range r2 = {4, 3, 2};
    Range r3 = {0, 4, 2};
    std::vector<size_t> q1 = {1, 1};
    std::vector<Range> q2 = {r1, r2, r3};
    Shape s1(v);
    auto t1 = Tensor::arange(s1, 0);
    std::cout << "Original:" << std::endl << *t1 << std::endl;
    auto t2 = t1->at(q1);
    Shape s2({1, 1, 4});
    real d2[] = {16, 17, 18, 19};
    auto x2 = Tensor::fromArr(s2, d2);
    testTemplate(*t2, *x2);
    auto t3 = t1->at(q2);
    std::cout << *t3 << std::endl;
    ASSERT_EQ(t3->isEmpty(), true);
}

TEST(TensorTestFixture, indexTensor2) {
    std::cout << std::endl << "Indexing tensor 2:" << std::endl;
    std::vector<size_t> v = {2, 3, 4, 5};
    Shape s1(v);
    auto t1 = Tensor::arange(s1, 0);
    std::cout << "Original:" << std::endl << *t1 << std::endl;
    Range r1 = {1, 2, 2};
    Range r2 = {1, 3, 2};
    Range r3 = {1, 4, 2};
    Range r4 = {1, 5, 1};
    std::vector q1 = {r1, r2, r3, r4};
    auto t2 = t1->at(q1);
    Shape s2({1, 1, 2, 4});
    real d2[] = {86, 87, 88, 89, 96, 97, 98, 99};
    auto x2 = Tensor::fromArr(s2, d2);
    testTemplate(*t2, *x2);
    Range r5 = {0, 1, 1};
    Range r6 = {0, 1, 1};
    Range r7 = {0, 2, 1};
    Range r8 = {1, 4, 2};
    std::vector q3 = {r5, r6, r7, r8};
    auto t3 = t2->at(q3);
    Shape s3({1, 1, 2, 2});
    real d3[] = {87, 89, 97, 99};
    auto x3 = Tensor::fromArr(s3, d3);
    testTemplate(*t3, *x3);
}

TEST(TensorTestFixture, sumTensor) {
    std::cout << std::endl << "Summing tensor:" << std::endl;
    std::vector<size_t> v = {1, 2, 12};
    Shape s1(v);
    auto t1 = Tensor::arange(s1, 0, 1);
    std::cout << "Original:" << std::endl << *t1 << std::endl;
    auto t2 = t1->sum();
    Shape s2({1});
    real data[] = {276};
    auto x2 = Tensor::fromArr(s2, data);
    testTemplate(*t2, *x2);
    t2->backward();
    auto g1 = Tensor::fromConst(s1, 1.);
    testTemplate(*t1->getGrad(), *g1);
}

TEST(TensorTestFixture, permuteShape) {
    std::cout << std::endl << "Permute tensor 1:" << std::endl;
    std::vector<size_t> v = {2, 3, 4};
    Shape s1(v);
    auto t1 = Tensor::arange(s1, 0);
    std::cout << "Original:" << std::endl << *t1 << std::endl;
    auto t2 = t1->perm({2, 1, 0});
    Shape s2({4, 3, 2});
    real d2[] = {0, 12, 4, 16, 8, 20, 1, 13, 5, 17, 9, 21, 2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23};
    auto x2 = Tensor::fromArr(s2, d2);
    testTemplate(*t2, *x2);
}

TEST(TensorTestFixture, indexPermTensor1) {
    std::cout << std::endl << "Indexing and permutating tensor 1:" << std::endl;
    std::vector<size_t> v = {2, 3, 4};
    Range r1 = {0, 2, 1};
    Range r2 = {1, 3, 2};
    Range r3 = {0, 4, 2};
    std::vector<Range> q2 = {r1, r2, r3};
    Shape s1(v);
    auto t1 = Tensor::arange(s1, 0);
    std::cout << "Orginal:" << std::endl << *t1 << std::endl;
    auto t2 = t1->at(q2);
    Shape s2({2, 1, 2});
    real d2[] = {4, 6, 16, 18};
    auto x2 = Tensor::fromArr(s2, d2);
    testTemplate(*t2, *x2);
    auto t3 = t2->perm({2, 1, 0});
    real d3[] = {4, 16, 6, 18};
    auto x3 = Tensor::fromArr(s2, d3);
    testTemplate(*t3, *x3);
}
