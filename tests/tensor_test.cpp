//
// Created by Trung Luu on 7/16/24.
//

#include "gtest/gtest.h"
#include "tensor.h"
#include "tensor_graph.h"

using namespace Toygrad::Tensor;

class TensorTestFixture : public testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {
    }
};

void assertEqTemplate(Tensor &actual, Tensor &expected) {
    std::cout << "Actual:" << std::endl << actual << std::endl;
    std::cout << "Expected:" << std::endl << expected << std::endl;
    ASSERT_EQ(actual, expected);
}

TEST(TensorTestFixture, indexTensor1) {
    std::cout << std::endl << "Indexing tensor 1:" << std::endl;
    Range r1 = {0, 2, 1};
    Range r2 = {4, 3, 2};
    Range r3 = {0, 4, 2};
    auto t1 = Tensor::arange({2, 3, 4}, 0);
    auto t2 = t1->at({1, 1});
    auto t3 = t1->at({r1, r2, r3});
    auto graph1 = std::make_unique<TensorGraph>(t2.get());
    auto graph2 = std::make_unique<TensorGraph>(t3.get());
    graph1->forward();
    graph2->forward();
    std::cout << "Original:" << std::endl << *t1 << std::endl;
    std::cout << *t3 << std::endl;
    Shape s2({4});
    real d2[] = {16, 17, 18, 19};
    auto x2 = Tensor::fromArr(s2, d2);
    x2->forward();
    assertEqTemplate(*t2, *x2);
    ASSERT_EQ(t3->isEmpty(), true);
}

TEST(TensorTestFixture, indexTensor2) {
    std::cout << std::endl << "Indexing tensor 2:" << std::endl;
    auto t1 = Tensor::arange({2, 3, 4, 5}, 0);
    Range r1 = {1, 2, 2};
    Range r2 = {1, 3, 2};
    Range r3 = {1, 4, 2};
    Range r4 = {1, 5, 1};
    auto t2 = t1->at({r1, r2, r3, r4});
    Range r5 = {0, 1, 1};
    Range r6 = {0, 1, 1};
    Range r7 = {0, 2, 1};
    Range r8 = {1, 4, 2};
    auto t3 = t2->at({r5, r6, r7, r8});
    auto graph = std::make_unique<TensorGraph>(t3.get());
    graph->forward();
    std::cout << "Original:" << std::endl << *t1 << std::endl;
    Shape s2({1, 1, 2, 4});
    real d2[] = {86, 87, 88, 89, 96, 97, 98, 99};
    auto x2 = Tensor::fromArr(s2, d2);
    x2->forward();
    assertEqTemplate(*t2, *x2);
    Shape s3({1, 1, 2, 2});
    real d3[] = {87, 89, 97, 99};
    auto x3 = Tensor::fromArr(s3, d3);
    x3->forward();
    assertEqTemplate(*t3, *x3);
}

TEST(TensorTestFixture, sumTensor1) {
    std::cout << std::endl << "Summing tensor 1:" << std::endl;
    Shape s1({1, 2, 12});
    auto t1 = Tensor::arange(s1, 0, 1);
    auto t2 = t1->sum();
    auto graph = std::make_unique<TensorGraph>(t2.get());
    graph->forward();
    graph->backward();
    std::cout << "Original:" << std::endl << *t1 << std::endl;
    real data[] = {276};
    auto x2 = Tensor::fromArr({1}, data);
    x2->forward();
    assertEqTemplate(*t2, *x2);
    auto g1 = Tensor::fromConst(s1, 1.);
    g1->forward();
    assertEqTemplate(*t1->getGrad(), *g1);
}

TEST(TensorTestFixture, sumTensor2) {
    std::cout << std::endl << "Summing tensor 2:" << std::endl;
    auto t1 = Tensor::arange({2, 3, 4}, 0, 1);
    auto t2 = t1->sum(1);
    auto t3 = t2->sum();
    auto graph = std::make_unique<TensorGraph>(t3.get());
    graph->forward();
    graph->backward();
    std::cout << "Original:" << std::endl << *t1 << std::endl;
    real data[] = {12, 15, 18, 21, 48, 51, 54, 57};
    auto x2 = Tensor::fromArr({2, 4}, data);
    x2->forward();
    assertEqTemplate(*t2, *x2);
}

TEST(TensorTestFixture, sumTensor3) {
    std::cout << std::endl << "Summing tensor 3:" << std::endl;
    auto t1 = Tensor::arange({2, 3, 4, 5}, 0, 1);
    auto t2 = t1->sum(2);
    auto graph = std::make_unique<TensorGraph>(t2.get());
    graph->forward();
    std::cout << "Original:" << std::endl << *t1 << std::endl;
    real data[] = {
        30.0, 34.0, 38.0, 42.0, 46.0, 110.0, 114.0, 118.0, 122.0, 126.0, 190.0, 194.0, 198.0, 202.0, 206.0, 270.0,
        274.0, 278.0, 282.0, 286.0, 350.0, 354.0, 358.0, 362.0, 366.0, 430.0, 434.0, 438.0, 442.0, 446.0
    };
    auto x2 = Tensor::fromArr({2, 3, 5}, data);
    x2->forward();
    assertEqTemplate(*t2, *x2);
}

void maxHelper(const TensorPtr &t1, const TensorPtr &x2, const TensorPtr &g1) {
    auto t2 = t1->max(1);
    auto t3 = t2->sum();
    auto graph = std::make_unique<TensorGraph>(t3.get());
    graph->forward();
    graph->backward();
    std::cout << "Original:" << std::endl << *t1 << std::endl;
    x2->forward();
    assertEqTemplate(*t2, *x2);
    g1->forward();
    assertEqTemplate(*t1->getGrad(), *g1);
}

TEST(TensorTestFixture, maxTensor1) {
    std::cout << std::endl << "Maxing tensor 1:" << std::endl;
    Shape s1({1, 2, 12});
    std::vector<real> data1 = {
        88, 99, 8, 35, 6, 54, 98, 67, 33, 93, 32, 1, 80, 95, 17, 72, 36, 41, 29, 1, 12, 87, 66, 43
    };
    auto t1 = Tensor::fromVec(s1, data1);
    real data2[] = {88, 99, 17, 72, 36, 54, 98, 67, 33, 93, 66, 43};
    auto x2 = Tensor::fromArr({1, 12}, data2);
    real data3[] = {1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1};
    auto g1 = Tensor::fromArr({1, 2, 12}, data3);
    maxHelper(t1, x2, g1);
}

TEST(TensorTestFixture, maxTensor2) {
    std::cout << std::endl << "Maxing tensor 2:" << std::endl;
    Shape s1({2, 3, 4});
    std::vector<real> data1 = {
        96., 53., 94., 9., 90., 18., 27., 19., 81., 85., 89., 94.,
        1., 15., 93., 0., 84., 8., 3., 92., 64., 45., 95., 48.
    };
    auto t1 = Tensor::fromVec(s1, data1);
    real data2[] = {96., 85., 94., 94., 84., 45., 95., 92.};
    auto x2 = Tensor::fromArr({2, 4}, data2);
    real data3[] = {1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0};
    auto g1 = Tensor::fromArr({2, 3, 4}, data3);
    maxHelper(t1, x2, g1);
}

TEST(TensorTestFixture, maxTensor3) {
    std::cout << std::endl << "Maxing tensor 3:" << std::endl;
    Shape s1({2, 3, 4});
    std::vector<real> data1 = {
        -1.0438, -1.2152, 1.0221, 0.0760, -0.3217, -0.0919, 1.6960, -0.7410, -1.5835, 1.1612, 0.0114, -1.1448,
        -0.7623, 0.6939, 0.3728, 0.0319, 1.6434, -0.6354, 0.8437, -0.3766, -0.4063, -2.9024, -0.5363, -1.0747
    };
    auto t1 = Tensor::fromVec(s1, data1);
    real data2[] = {-0.3217, 1.1612, 1.6960, 0.0760, 1.6434, 0.6939, 0.8437, 0.0319};
    auto x2 = Tensor::fromArr({2, 4}, data2);
    real data3[] = {0., 0., 0., 1., 1., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 1., 1., 0., 1., 0., 0., 0., 0., 0.};
    auto g1 = Tensor::fromArr({2, 3, 4}, data3);
    maxHelper(t1, x2, g1);
}

TEST(TensorTestFixture, permTensor1) {
    std::cout << std::endl << "Permute shape 1:" << std::endl;
    auto t1 = Tensor::arange({2, 3, 4}, 0);
    auto t2 = t1->perm({2, 1, 0});
    auto graph = std::make_unique<TensorGraph>(t2.get());
    graph->forward();
    std::cout << "Original:" << std::endl << *t1 << std::endl;
    real d2[] = {0, 12, 4, 16, 8, 20, 1, 13, 5, 17, 9, 21, 2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23};
    auto x2 = Tensor::fromArr({4, 3, 2}, d2);
    x2->forward();
    assertEqTemplate(*t2, *x2);
}

TEST(TensorTestFixture, indexPermTensor1) {
    std::cout << std::endl << "Indexing and permutating tensor 1:" << std::endl;
    Range r1 = {0, 2, 1};
    Range r2 = {1, 3, 2};
    Range r3 = {0, 4, 2};
    auto t1 = Tensor::arange({2, 3, 4}, 0);
    auto t2 = t1->at({r1, r2, r3});
    auto t3 = t2->perm({2, 1, 0});
    auto graph = std::make_unique<TensorGraph>(t3.get());
    graph->forward();
    std::cout << "Original:" << std::endl << *t1 << std::endl;
    Shape s2({2, 1, 2});
    real d2[] = {4, 6, 16, 18};
    auto x2 = Tensor::fromArr(s2, d2);
    x2->forward();
    assertEqTemplate(*t2, *x2);
    real d3[] = {4, 16, 6, 18};
    auto x3 = Tensor::fromArr(s2, d3);
    x3->forward();
    assertEqTemplate(*t3, *x3);
}

// TEST(TensorTestFixture, indexPermTensor2) {
//     std::cout << std::endl << "Indexing and permutating tensor 1:" << std::endl;
//     std::vector<size_t> v = {2, 3, 4};
//     Range r1 = {0, 2, 1};
//     Range r2 = {1, 3, 2};
//     Range r3 = {0, 4, 2};
//     std::vector q2 = {r1, r2, r3};
//     Shape s1(v);
//     auto t1 = Tensor::arange(s1, 0);
//     auto t2 = t1->at(q2);
//     Shape s2({2, 1, 2});
//     auto t3 = t2->perm({2, 1, 0});
//     Shape s3({1});
//     auto t5 = Tensor::fromConst(s3, 1.0);
//     auto t4 = t3->at(1)->at(0)->at(1)->broadcastTo(s3)->add(t5);
//     // real d4[] = {18};
//     // std::vector<size_t> indices;
//     // Shape s4(indices);
//     // auto x4 = Tensor::fromArr(s4, d4);
//     // assertEqTemplate(*t4, *x4);
//     std::cout << *t4 << std::endl;
// }

TEST(TensorTestFixture, indexSumTensor1) {
    std::cout << std::endl << "Indexing and summing tensor 1:" << std::endl;
    Range r1 = {0, 2, 1};
    Range r2 = {1, 3, 2};
    Range r3 = {0, 4, 2};
    auto t1 = Tensor::arange({2, 3, 4}, 0);
    auto t2 = t1->at({r1, r2, r3})->sum(1);
    auto graph = std::make_unique<TensorGraph>(t2.get());
    graph->forward();
    std::cout << "Original:" << std::endl << *t1 << std::endl;
    real d2[] = {4, 6, 16, 18};
    auto x2 = Tensor::fromArr({2, 2}, d2);
    x2->forward();
    assertEqTemplate(*t2, *x2);
}

TEST(TensorTestFixture, sumTensorGrad1) {
    std::cout << std::endl << "Sum tensor's gradient 1:" << std::endl;
    Shape s1({2, 3, 4});
    auto t1 = Tensor::arange(s1, 0, 1);
    auto t2 = t1->sum(1);
    auto t3 = t2->sum();
    auto graph = std::make_unique<TensorGraph>(t3.get());
    graph->forward();
    graph->backward();
    std::cout << "Original:" << std::endl << *t1 << std::endl;
    auto g1 = Tensor::fromConst(s1, 1.);
    g1->forward();
    assertEqTemplate(*t1->getGrad(), *g1);
}

TEST(TensorTestFixture, softmaxTensor1) {
    std::cout << std::endl << "Softmax tensor 1:" << std::endl;
    auto t1 = Tensor::arange({2, 3, 4}, 0, 1);
    auto t2 = t1->softmax();
    auto graph = std::make_unique<TensorGraph>(t2.get());
    graph->forward();
    std::cout << "Original:" << std::endl << *t1 << std::endl;
    std::cout << "Softmax:" << std::endl << *t2 << std::endl;
}

TEST(TensorTestFixture, softmaxTensor2) {
    std::cout << std::endl << "Softmax tensor 2:" << std::endl;
    auto t1 = Tensor::arange({2, 3, 4}, 0, 1);
    auto t2 = t1->softmax(1);
    auto graph = std::make_unique<TensorGraph>(t2.get());
    graph->forward();
    std::cout << "Original:" << std::endl << *t1 << std::endl;
    std::cout << "Softmax:" << std::endl << *t2 << std::endl;
}

TEST(TensorTestFixture, softmaxTensor3) {
    std::cout << std::endl << "Softmax tensor 3:" << std::endl;
    Range r1 = {0, 2, 1};
    Range r2 = {1, 3, 2};
    Range r3 = {0, 4, 2};
    auto t1 = Tensor::arange({2, 3, 4}, 0);
    auto t2 = t1->at({r1, r2, r3})->softmax(0);
    auto graph = std::make_unique<TensorGraph>(t2.get());
    graph->forward();
    std::cout << "Original:" << std::endl << *t1 << std::endl;
    std::cout << "Softmax:" << std::endl << *t2 << std::endl;
}

TEST(TensorTestFixture, softmax4) {
    std::cout << std::endl << "Softmax tensor 4:" << std::endl;
    Range r1 = {0, 2, 1};
    Range r2 = {1, 3, 2};
    Range r3 = {0, 4, 2};
    auto t1 = Tensor::arange({2, 3, 4}, 0);
    auto t2 = t1->at({r1, r2, r3})->perm({1, 2, 0})->softmax(1);
    auto graph = std::make_unique<TensorGraph>(t2.get());
    graph->forward();
    std::cout << "Original:" << std::endl << *t1 << std::endl;
    std::cout << "Softmax:" << std::endl << *t2 << std::endl;
}

void broadcastTensorHelper(const TensorPtr &t1, const std::vector<size_t> &v2, real *d2) {
    auto t2 = t1->broadcastTo(v2);
    auto graph = std::make_unique<TensorGraph>(t2.get());
    graph->forward();
    std::cout << "Original:" << std::endl << *t1 << std::endl;
    auto x2 = Tensor::fromArr(v2, d2);
    x2->forward();
    assertEqTemplate(*t2, *x2);
}

TEST(TensorTestFixture, broadcastTensor1) {
    std::cout << std::endl << "Broadcast tensor 1:" << std::endl;
    auto t1 = Tensor::arange(Shape({2, 1, 4}), 0);
    std::vector<size_t> v2 = {2, 3, 4};
    real d2[] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7};
    broadcastTensorHelper(t1, v2, d2);

    auto t3 = Tensor::arange(Shape({1, 1, 4}), 0);
    std::vector<size_t> v4 = {2, 3, 4};
    real d4[] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    broadcastTensorHelper(t3, v4, d4);

    auto t5 = Tensor::arange(Shape({1, 3, 1}), 0);
    std::vector<size_t> v6 = {2, 3, 4};
    real d6[] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
    broadcastTensorHelper(t5, v6, d6);

    auto t7 = Tensor::arange(Shape({2, 1, 1}), 0);
    std::vector<size_t> v8 = {2, 3, 4};
    real d8[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    broadcastTensorHelper(t7, v8, d8);

    auto t9 = Tensor::arange(Shape({2, 4, 1}), 0);
    std::vector<size_t> v10 = {2, 4, 3};
    real d10[] = {0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7};
    broadcastTensorHelper(t9, v10, d10);
}

TEST(TensorTestFixture, broadcastTensor2) {
    std::cout << std::endl << "Broadcast tensor 2:" << std::endl;
    auto t1 = Tensor::arange(Shape({1, 4}), 0);
    std::vector<size_t> v2 = {2, 3, 4};
    real d2[] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    broadcastTensorHelper(t1, v2, d2);

    auto t3 = Tensor::arange(Shape({1}), 11);
    std::vector<size_t> v4 = {3, 4};
    real d4[] = {11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11};
    broadcastTensorHelper(t3, v4, d4);
}

TEST(TensorTestFixture, broadcastTensor3) {
    std::cout << std::endl << "Broadcast tensor 3:" << std::endl;
    auto t1 = Tensor::arange({2, 4}, 0);
    t1->forward();
    std::cout << "Original:" << std::endl << *t1 << std::endl;
    ASSERT_EQ(false, t1->isBroadcastableTo({2, 3, 4}));

    auto t2 = Tensor::arange({4}, 0);
    t2->forward();
    std::cout << "Original:" << std::endl << *t2 << std::endl;
    ASSERT_EQ(true, t2->isBroadcastableTo({2, 3, 4}));

    auto t3 = Tensor::arange({1, 2, 3, 4}, 0);
    t3->forward();
    std::cout << "Original:" << std::endl << *t3 << std::endl;
    ASSERT_EQ(false, t3->isBroadcastableTo({2, 3, 4}));
}

void squeezeHelper(const TensorPtr &t1, int64_t dim, const std::vector<size_t> &v2, real *d2) {
    auto t2 = t1->squeeze(dim);
    auto graph = std::make_unique<TensorGraph>(t2.get());
    graph->forward();
    std::cout << "Original:" << std::endl << *t1 << std::endl;
    auto x2 = Tensor::fromArr(v2, d2);
    x2->forward();
    assertEqTemplate(*t2, *x2);
}

TEST(TensorTestFixture, squeeze1) {
    std::cout << std::endl << "Squeeze 1:" << std::endl;

    auto t1 = Tensor::arange({2, 1, 1, 4, 1}, 0);
    std::vector<size_t> v2 = {2, 4};
    real d2[] = {0, 1, 2, 3, 4, 5, 6, 7};
    squeezeHelper(t1, -1, v2, d2);

    auto t3 = Tensor::arange({3, 4, 1}, 0);
    std::vector<size_t> v4 = {3, 4};
    real d4[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    squeezeHelper(t3, 2, v4, d4);

    auto t5 = Tensor::arange({3, 4}, 0);
    std::vector<size_t> v6 = {3, 4};
    real d6[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    squeezeHelper(t5, 1, v6, d6);
}

TEST(TensorTestFixture, squeeze2) {
    std::cout << std::endl << "Squeeze 2:" << std::endl;

    auto t1 = Tensor::arange({1}, 0);
    t1->forward();
    std::cout << "Original:" << std::endl << *t1 << std::endl;
    ASSERT_EQ(false, t1->isSqueezable());

    auto t2 = Tensor::arange({2, 4, 1}, 0);
    t2->forward();
    std::cout << "Original:" << std::endl << *t2 << std::endl;
    ASSERT_EQ(true, t2->isSqueezable(2));

    auto t3 = Tensor::arange({1, 1}, 0);
    t3->forward();
    std::cout << "Original:" << std::endl << *t3 << std::endl;
    ASSERT_EQ(false, t3->isSqueezable());

    auto t4 = Tensor::arange({1, 1}, 0);
    t4->forward();
    std::cout << "Original:" << std::endl << *t4 << std::endl;
    ASSERT_EQ(true, t3->isSqueezable(0));
}

void unsqueezeHelper(const TensorPtr &t1, int64_t dim, const std::vector<size_t> &v2, real *d2) {
    auto t2 = t1->unsqueeze(dim);
    auto graph = std::make_unique<TensorGraph>(t2.get());
    graph->forward();
    std::cout << "Original:" << std::endl << *t1 << std::endl;
    auto x2 = Tensor::fromArr(v2, d2);
    x2->forward();
    assertEqTemplate(*t2, *x2);
}

TEST(TensorTestFixture, unsqueeze1) {
    std::cout << std::endl << "Unsqueeze 1:" << std::endl;

    auto t1 = Tensor::arange({2, 3, 4}, 0);
    std::vector<size_t> v2 = {2, 3, 4, 1};
    real d2[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    unsqueezeHelper(t1, -1, v2, d2);

    auto t3 = Tensor::arange({2, 3, 4}, 0);
    std::vector<size_t> v4 = {2, 1, 3, 4};
    real d4[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    unsqueezeHelper(t3, 1, v4, d4);
}

void matmulHelper(const std::vector<size_t> &v1, real start1, const std::vector<size_t> &v2, real start2,
                  const real *d3) {
    auto t1 = Tensor::arange(v1, start1);
    auto t2 = Tensor::arange(v2, start2);
    auto t3 = t1->matmul(t2);
    auto graph = std::make_unique<TensorGraph>(t3.get());
    graph->forward();
    std::cout << "Matrix 1:" << std::endl << *t1 << std::endl;
    std::cout << "Matrix 2:" << std::endl << *t2 << std::endl;
    std::vector<size_t> v3 = v1;
    v3[v3.size() - 1] = v2[v2.size() - 1];
    auto x3 = Tensor::fromArr(v3, d3);
    x3->forward();
    assertEqTemplate(*t3, *x3);
}

TEST(TensorTestFixture, matmul1) {
    std::cout << std::endl << "Matmul 1:" << std::endl;
    real d1[] = {20, 23, 26, 29, 56, 68, 80, 92};
    real d2[] = {301, 322, 343, 364, 697, 754, 811, 868, 1093, 1186, 1279, 1372};
    real d3[] = {67.5};
    matmulHelper({2, 3}, 0, {3, 4}, 0, d1);
    matmulHelper({3, 6}, 1, {6, 4}, 1, d2);
    matmulHelper({1, 1}, 9, {1, 1}, 7.5, d3);
}

TEST(TensorTestFixture, matmul2) {
    std::cout << std::endl << "Matmul 2:" << std::endl;
    real d1[] = {20, 23, 26, 29, 56, 68, 80, 92, 344, 365, 386, 407, 488, 518, 548, 578};
    real d2[] = {
        301, 322, 343, 364, 697, 754, 811, 868, 1093, 1186, 1279, 1372,
        4585, 4714, 4843, 4972, 5845, 6010, 6175, 6340, 7105, 7306, 7507, 7708,
        14053, 14290, 14527, 14764, 16177, 16450, 16723, 16996, 18301, 18610, 18919, 19228
    };
    real d3[] = {15, 20, 25, 18, 24, 30};
    matmulHelper({2, 2, 3}, 0, {2, 3, 4}, 0, d1);
    matmulHelper({3, 1, 3, 6}, 1, {3, 1, 6, 4}, 1, d2);
    matmulHelper({1, 2, 1}, 5, {1, 1, 3}, 3, d3);
}

TEST(TensorTestFixture, matmul3) {
    std::cout << std::endl << "Matmul 3:" << std::endl;
    real d1[] = {
        41., 63., 68., 83., 5., 59., 95., 99., 47., 13., 57., 77., 6., 0., 28., 57., 0., 96., 25., 16., 84., 88., 54.,
        5.
    };
    auto t1 = Tensor::fromArr({2, 3, 4}, d1);
    real d2[] = {22., 31., 7., 55., 36., 27., 72., 3., 86., 90., 85., 66., 95., 12., 7., 93.};
    auto t2 = Tensor::fromArr({2, 4, 2}, d2);
    auto t3 = t1->matmul(t2);
    auto t4 = t3->sum();
    auto graph = std::make_unique<TensorGraph>(t4.get());
    graph->forward();
    graph->backward();
    real d3[] = {9767., 6821., 11071., 6262., 8721., 3942., 3575., 6177., 10647., 8124., 19869., 14481.};
    auto x3 = Tensor::fromArr({2, 3, 2}, d3);
    x3->forward();
    assertEqTemplate(*t3, *x3);
    real d4[] = {
        53., 62., 63., 75., 53., 62., 63., 75., 53., 62., 63., 75., 176., 151., 107., 100., 176., 151., 107., 100.,
        176., 151., 107., 100.
    };
    auto g1 = Tensor::fromArr({2, 3, 4}, d4);
    g1->forward();
    assertEqTemplate(*t1->getGrad(), *g1);
    real d5[] = {93., 93., 135., 135., 220., 220., 259., 259., 90., 90., 184., 184., 107., 107., 78., 78.};
    auto g2 = Tensor::fromArr({2, 4, 2}, d5);
    g2->forward();
    assertEqTemplate(*t2->getGrad(), *g2);
}
