/* Copyright 2019-2021 Canaan Inc.
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

using namespace nncase;
using namespace nncase::runtime;
using namespace ortki;

class StackTest : public KernelTest,
                  public ::testing::TestWithParam<
                      std::tuple<nncase::typecode_t, dims_t, int64_t>> {
  public:
    void SetUp() override {
        auto &&[typecode, l_shape, value] = GetParam();

        input =
            hrt::create(typecode, l_shape, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
        init_tensor(input);

        axes_value = value;
    }

    void TearDown() override {}

  protected:
    runtime_tensor input;
    int64_t axes_value;
};

INSTANTIATE_TEST_SUITE_P(
    Stack, StackTest,
    testing::Combine(testing::Values(dt_float32, dt_int32, dt_int16, dt_float64,
                                     dt_int8, dt_uint8, dt_uint16, dt_uint32,
                                     dt_uint64, dt_int64, dt_bfloat16,
                                     dt_float16, dt_boolean),
                     testing::Values(dims_t{
                         1} /*, dims_t{2}, dims_t{1,
                   1}, dims_t{1, 2, 4, 8}, dims_t{4,
                   4, 8}*/),
                     testing::Values(0, -1)));

TEST_P(StackTest, Stack) {

    // actual
    value_t field1 = input.impl();
    std::vector<value_t> fields;
    fields.push_back(field1);
    auto output_tuple = tuple(std::in_place, std::move(fields));
    int64_t axes_array[] = {axes_value};
    auto axes = hrt::create(dt_int64, {1},
                            {reinterpret_cast<gsl::byte *>(axes_array),
                             sizeof(axes_array)},
                            true, host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");
    auto output = kernels::stackvm::stack(output_tuple, axes.impl())
                      .expect("stack failed");
    runtime_tensor actual(output.as<tensor>().expect("as tensor failed"));

    int64_t output_shape_array[] = {1, 1};
    auto output_shape =
        hrt::create(dt_int64, {2},
                    {reinterpret_cast<gsl::byte *>(output_shape_array),
                     sizeof(output_shape_array)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    auto expected1 =
        kernels::stackvm::reshape(input.impl(), output_shape.impl())
            .expect("stack failed");
    runtime_tensor expected(expected1.as<tensor>().expect("as tensor failed"));

    bool result = is_same_tensor(expected, actual) ||
                  cosine_similarity_tensor(expected, actual);

    if (!result) {
        std::cout << "actual ";
        print_runtime_tensor(actual);
        std::cout << "expected ";
        print_runtime_tensor(expected);
    }

    // compare
    EXPECT_TRUE(result);
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}