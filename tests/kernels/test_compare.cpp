/* Copyright 2019-2023 Canaan Inc.
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
#include "kernel_test.h"
#include <gtest/gtest.h>
#include <iostream>
#include <nncase/kernels/stackvm/tensor_ops.h>
#include <nncase/runtime/datatypes.h>
#include <nncase/runtime/runtime_tensor.h>
#include <nncase/runtime/simple_types.h>
#include <nncase/runtime/stackvm/opcode.h>
#include <ortki/operators.h>

#define ORTKI_OP_1(operand_a, operand_b, op_a) op_a(operand_a, operand_b)
#define ORTKI_OP_2(operand_a, operand_b, op_a, op_b) op_a(op_b(operand_a, operand_b))
#define ORTKI_OP(num, operand_a, operand_b, ...) ORTKI_OP_##num(operand_a, operand_b, __VA_ARGS__)

#define READY_INPUT() \
    auto l_ort = runtime_tensor_2_ort_tensor(lhs); \
    auto r_ort = runtime_tensor_2_ort_tensor(rhs);

#define GET_EXPECT(ortop_num, ...) \
    auto output_ort = ORTKI_OP(ortop_num, l_ort, r_ort, __VA_ARGS__); \
    size_t size = 0;\
    void *ptr_ort = tensor_buffer(output_ort, &size);\
    dims_t shape(tensor_rank(output_ort));\
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));\
    auto expected = hrt::create(dt_boolean, shape,\
                                {reinterpret_cast<gsl::byte *>(ptr_ort), size},\
                                true, host_runtime_tensor::pool_cpu_only)\
                        .expect("create tensor failed");

#define GET_ACTUAL(op_name) \
    auto output = kernels::stackvm::compare( \
                      op_name, \
                      lhs.impl(), rhs.impl()) \
                      .expect("compare failed"); \
    runtime_tensor actual(output.as<tensor>().expect("as tensor failed"));

#define CHECK_RESULT() \
    bool result = is_same_tensor(expected, actual) || \
                  cosine_similarity_tensor(expected, actual); \
    if (!result) { \
        print_runtime_tensor(actual); \
        print_runtime_tensor(expected); \
    } \
    EXPECT_TRUE(result);

#define _COMPARE_BODY(sub_op_name, ortki_op_num, ...) \
    READY_INPUT() \
    GET_EXPECT(ortki_op_num, __VA_ARGS__) \
    GET_ACTUAL(sub_op_name) \
    CHECK_RESULT()

#define COMPARE_BODY(test_name, sub_op_name, ortki_op_num, ...) \
    TEST_P(CompareTest, test_name) { \
    _COMPARE_BODY(sub_op_name, ortki_op_num, __VA_ARGS__) \
}

using namespace nncase;
using namespace nncase::runtime;
using namespace ortki;
using namespace nncase::runtime::stackvm;

class CompareTest : public KernelTest,
                    public ::testing::TestWithParam<std::tuple<nncase::typecode_t, dims_t, dims_t>> {
  public:
    void SetUp() override {
        auto &&[typecode, l_shape, r_shape] = GetParam();

        lhs = hrt::create(typecode, l_shape, host_runtime_tensor::pool_cpu_only)
                  .expect("create tensor failed");
        rhs = hrt::create(typecode, r_shape, host_runtime_tensor::pool_cpu_only)
                  .expect("create tensor failed");
        init_tensor(lhs);
        init_tensor(rhs);
    }

    void TearDown() override {}

  protected:
    runtime_tensor lhs;
    runtime_tensor rhs;
};

INSTANTIATE_TEST_SUITE_P(
    compare, CompareTest,
    testing::Combine(testing::Values(dt_boolean, dt_int64, dt_int32),
                     testing::Values(dims_t{1, 3, 16, 16},
                                     dims_t{1, 1, 16, 16}),
                     testing::Values(dims_t{1}, dims_t{16}, dims_t{1, 16},
                                     dims_t{1, 16, 16}, dims_t{3, 3, 1, 16})));

COMPARE_BODY(not_equal, compare_op_t::not_equal, 2, ortki_Not, ortki_Equal)
COMPARE_BODY(equal, compare_op_t::equal, 1, ortki_Equal)
COMPARE_BODY(greater_or_equal, compare_op_t::greater_or_equal, 1, ortki_GreaterOrEqual)
COMPARE_BODY(greater_than, compare_op_t::greater_than, 1, ortki_Greater)
COMPARE_BODY(lower_or_equal, compare_op_t::lower_or_equal, 1, ortki_LessOrEqual)
COMPARE_BODY(lower_than, compare_op_t::lower_than, 1, ortki_Less)

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}