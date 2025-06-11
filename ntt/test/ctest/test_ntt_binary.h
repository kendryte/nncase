/* Copyright 2019-2024 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "nncase/ntt/shape.h"
#include "nncase/ntt/tensor.h"
#include "ntt_test.h"
#include "ortki_helper.h"
#include <gtest/gtest.h>
#include <iostream>
#include <nncase/ntt/ntt.h>
#include <ortki/operators.h>
#include <string_view>

using namespace nncase;
using namespace ortki;

#define _TEST_VECTOR(T, lmul)                                                  \
    test_vector<T, (NTT_VLEN) / (sizeof(T) * 8) * lmul>();

#define TEST_VECTOR(T)                                                         \
    _TEST_VECTOR(T, 1)                                                         \
    _TEST_VECTOR(T, 2)                                                         \
    _TEST_VECTOR(T, 4)                                                         \
    _TEST_VECTOR(T, 8)


#define DEFINE_test_vector(ntt_op, Ort_op)                                                  \
    template <typename T, size_t vl> void test_vector() {                      \
        ntt::vector<T, vl> ntt_lhs, ntt_rhs;                                   \
        NttTest::init_tensor(ntt_lhs, static_cast<T>(-10),                     \
                             static_cast<T>(10));                              \
        NttTest::init_tensor(ntt_rhs, static_cast<T>(-10),                     \
                             static_cast<T>(10));                              \
        auto ntt_output1 = ntt::ntt_op(ntt_lhs, ntt_rhs);                    \
        auto ort_lhs = NttTest::ntt2ort(ntt_lhs);                              \
        auto ort_rhs = NttTest::ntt2ort(ntt_rhs);                              \
        auto ort_output = ortki_##Ort_op(ort_lhs, ort_rhs);                    \
        ntt::vector<T, vl> ntt_output2;                                        \
        NttTest::ort2ntt(ort_output, ntt_output2);                             \
        EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));        \
    }                                                                          



#define PRINT_TENSOR(tensor)                                                  \
    nncase::ntt::apply(tensor.shape(), [&](auto index) {                       \
        printf("c[%ld, %ld] = %f\n", index[0], index[1],\
                tensor(index));                                       \
    });                                                                       \


#define DECLARE_NTT_TENSOR(element_type, tensor_name, shape_param)                                            \
    /* is fixed_shape */                                                        \
    auto ntt_tensor_##tensor_name =  make_unique_tensor<element_type>(shape_param); \
    NttTest::init_tensor(*ntt_tensor_##tensor_name, -10.f, 10.f)

#define BINARY_TEST_BODY(lhs_name, rhs_name, out_name, golden_name, ntt_op, ort_op) \
    /* ntt */ \
    ntt::binary<ntt::ops::ntt_op>(*ntt_tensor_##lhs_name, *ntt_tensor_##rhs_name, *ntt_tensor_##out_name); \
    /* ort */ \
    auto ort_lhs = NttTest::ntt2ort(*ntt_tensor_##lhs_name); \
    auto ort_rhs = NttTest::ntt2ort(*ntt_tensor_##rhs_name); \
    auto ort_output = ortki_##ort_op(ort_lhs, ort_rhs); \
    /* compare */ \
    NttTest::ort2ntt(ort_output, *ntt_tensor_##golden_name); \
    EXPECT_TRUE(NttTest::compare_tensor(*ntt_tensor_##out_name, *ntt_tensor_##golden_name));




#define GENERATE_BINARY_TEST(test_group, test_name, \
                            lhs_shape, rhs_shape, output_shape, \
                           element_type, ntt_op, ort_op) \
TEST(test_group, test_name) { \
                DECLARE_NTT_TENSOR(element_type, lhs, lhs_shape); \
                DECLARE_NTT_TENSOR(element_type, rhs, rhs_shape); \
                DECLARE_NTT_TENSOR(element_type, out, output_shape); \
                DECLARE_NTT_TENSOR(element_type, golden, output_shape); \
                BINARY_TEST_BODY(lhs, rhs, out, golden, ntt_op, ort_op) \
}


#define GET_SHAPE_fixed(...) (fixed_shape_v<__VA_ARGS__>)
#define GET_SHAPE_dynamic(...) (ntt::make_shape(__VA_ARGS__))

#define MAKE_SUFFIX_IMPL(s1, s2, s3, suffix_type) s1##_##s2##_##s3##_##suffix_type
#define MAKE_SUFFIX(s1, s2, s3, suffix_type) MAKE_SUFFIX_IMPL(s1, s2, s3, suffix_type)

#define GENERATE_BINARY_TEST_GROUP(TestNamePrefix, LhsShapeType, RhsShapeType, OutShapeType, DataType, Ntt_op, Ort_op) \
    GENERATE_BINARY_TEST(TestNamePrefix, MAKE_SUFFIX(LhsShapeType, RhsShapeType, OutShapeType, normal), \
                         GET_SHAPE_##LhsShapeType(1, 3, 16, 16), \
                         GET_SHAPE_##RhsShapeType(1, 3, 16, 16), \
                         GET_SHAPE_##OutShapeType(1, 3, 16, 16), \
                         DataType, Ntt_op, Ort_op) \
    GENERATE_BINARY_TEST(TestNamePrefix, MAKE_SUFFIX(LhsShapeType, RhsShapeType, OutShapeType, broadcast_lhs_scalar), \
                         GET_SHAPE_##LhsShapeType(1), \
                         GET_SHAPE_##RhsShapeType(1, 3, 16, 16), \
                         GET_SHAPE_##OutShapeType(1, 3, 16, 16), \
                         DataType, Ntt_op, Ort_op) \
    GENERATE_BINARY_TEST(TestNamePrefix, MAKE_SUFFIX(LhsShapeType, RhsShapeType, OutShapeType, broadcast_rhs_scalar), \
                         GET_SHAPE_##LhsShapeType(1, 3, 16, 16), \
                         GET_SHAPE_##RhsShapeType(1), \
                         GET_SHAPE_##OutShapeType(1, 3, 16, 16), \
                         DataType, Ntt_op, Ort_op) \
    GENERATE_BINARY_TEST(TestNamePrefix, MAKE_SUFFIX(LhsShapeType, RhsShapeType, OutShapeType, broadcast_lhs_vector), \
                         GET_SHAPE_##LhsShapeType(16), \
                         GET_SHAPE_##RhsShapeType(1, 3, 16, 16), \
                         GET_SHAPE_##OutShapeType(1, 3, 16, 16), \
                         DataType, Ntt_op, Ort_op) \
    GENERATE_BINARY_TEST(TestNamePrefix, MAKE_SUFFIX(LhsShapeType, RhsShapeType, OutShapeType, broadcast_rhs_vector), \
                         GET_SHAPE_##LhsShapeType(1, 3, 16, 16), \
                         GET_SHAPE_##RhsShapeType(16), \
                         GET_SHAPE_##OutShapeType(1, 3, 16, 16), \
                         DataType, Ntt_op, Ort_op) \
    GENERATE_BINARY_TEST(TestNamePrefix, MAKE_SUFFIX(LhsShapeType, RhsShapeType, OutShapeType, broadcast_multidirectional), \
                         GET_SHAPE_##LhsShapeType(1, 3, 1, 16), \
                         GET_SHAPE_##RhsShapeType(3, 1, 16, 1), \
                         GET_SHAPE_##OutShapeType(3, 3, 16, 16), \
                         DataType, Ntt_op, Ort_op)


