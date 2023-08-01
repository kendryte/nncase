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

class SliceTest
    : public KernelTest,
      public ::testing::TestWithParam<std::tuple<
          nncase::typecode_t, dims_t, dims_t, dims_t, dims_t, dims_t>> {
  public:
    void SetUp() override {
        auto &&[typecode, l_shape, value1, value2, value3, value4] = GetParam();

        int32_t input_array[120];

        for (int i = 0; i < 120; ++i) {
            input_array[i] = i;
        }

        input = hrt::create(typecode, l_shape,
                            {reinterpret_cast<gsl::byte *>(input_array),
                             sizeof(input_array)},
                            true, host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");

        size_t begin_size = value1.size();
        int64_t *begin_array = (int64_t *)malloc(begin_size * sizeof(int64_t));
        std::copy(value1.begin(), value1.end(), begin_array);
        begin = hrt::create(dt_int64, {begin_size},
                            {reinterpret_cast<gsl::byte *>(begin_array),
                             begin_size * sizeof(int64_t)},
                            true, host_runtime_tensor::pool_cpu_only)
                    .expect("create1 tensor failed");

        size_t end_size = value2.size();
        int64_t *end_array = (int64_t *)malloc(end_size * sizeof(int64_t));
        std::copy(value2.begin(), value2.end(), end_array);
        end = hrt::create(dt_int64, {begin_size},
                          {reinterpret_cast<gsl::byte *>(end_array),
                           end_size * sizeof(int64_t)},
                          true, host_runtime_tensor::pool_cpu_only)
                  .expect("create2 tensor failed");

        size_t axes_size = value3.size();
        int64_t *axes_array = (int64_t *)malloc(axes_size * sizeof(int64_t));
        std::copy(value3.begin(), value3.end(), axes_array);
        axes = hrt::create(dt_int64, {begin_size},
                           {reinterpret_cast<gsl::byte *>(axes_array),
                            axes_size * sizeof(int64_t)},
                           true, host_runtime_tensor::pool_cpu_only)
                   .expect("create3 tensor failed");

        size_t strides_size = value4.size();
        int64_t *strides_array =
            (int64_t *)malloc(strides_size * sizeof(int64_t));
        std::copy(value4.begin(), value4.end(), strides_array);
        strides = hrt::create(dt_int64, {begin_size},
                              {reinterpret_cast<gsl::byte *>(strides_array),
                               strides_size * sizeof(int64_t)},
                              true, host_runtime_tensor::pool_cpu_only)
                      .expect("create4 tensor failed");
    }

    void TearDown() override {}

  protected:
    runtime_tensor input;
    runtime_tensor begin;
    runtime_tensor end;
    runtime_tensor axes;
    runtime_tensor strides;
};

INSTANTIATE_TEST_SUITE_P(
    Slice, SliceTest,
    testing::Combine(testing::Values(dt_int32),
                     testing::Values(dims_t{2, 3, 4, 5}, dims_t{1, 4, 5, 6},
                                     dims_t{1, 1, 1, 120}, dims_t{2, 2, 5, 6},
                                     dims_t{1, 1, 2, 60}),
                     testing::Values(dims_t{0, 0, 0, 0}),
                     testing::Values(dims_t{1, 1, 1, 5}),
                     testing::Values(dims_t{0, 1, 2, 3}),
                     testing::Values(dims_t{1, 1, 1, 1})));

TEST_P(SliceTest, Slice) {

    // expected
    int32_t result[] = {0, 1, 2, 3, 4};
    auto expected =
        hrt::create(input.datatype(), {1, 1, 1, 5},
                    {reinterpret_cast<gsl::byte *>(result), sizeof(result)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    // actual
    auto output =
        kernels::stackvm::slice(input.impl(), begin.impl(), end.impl(),
                                axes.impl(), strides.impl())
            .expect("slice failed");
    runtime_tensor actual(output.as<tensor>().expect("as tensor failed"));

    bool result1 = is_same_tensor(expected, actual) ||
                   cosine_similarity_tensor(expected, actual);

    if (!result1) {
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