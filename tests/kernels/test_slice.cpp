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
      public ::testing::TestWithParam<std::tuple<nncase::typecode_t, dims_t>> {
  public:
    void SetUp() override {
        auto &&[typecode, l_shape] = GetParam();

        input =
            hrt::create(typecode, l_shape, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
        init_tensor(input);
    }

    void TearDown() override {}

  protected:
    runtime_tensor input;
};

INSTANTIATE_TEST_SUITE_P(Slice, SliceTest,
                         testing::Combine(testing::Values(dt_int32),
                                          testing::Values(dims_t{2, 3, 4, 5})));

TEST_P(SliceTest, Slice) {
    //    auto l_ort = runtime_tensor_2_ort_tensor(input);

    // expected
    //    size_t size = 0;
    int32_t result[] = {0, 1, 2, 3, 4};
    auto expected = hrt::create(input.datatype(), {1, 1, 1, 5},
                                {reinterpret_cast<gsl::byte *>(result), 20},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    // actual
    int32_t begin_array[] = {0, 0, 0, 0};
    auto begin = hrt::create(input.datatype(), {4},
                             {reinterpret_cast<gsl::byte *>(begin_array), 16},
                             true, host_runtime_tensor::pool_cpu_only)
                     .expect("create tensor failed");
    int32_t end_array[] = {1, 1, 1, 5};
    auto end = hrt::create(input.datatype(), {4},
                           {reinterpret_cast<gsl::byte *>(end_array), 16}, true,
                           host_runtime_tensor::pool_cpu_only)
                   .expect("create tensor failed");
    int32_t axes_array[] = {0, 1, 2, 3};
    auto axes = hrt::create(input.datatype(), {4},
                            {reinterpret_cast<gsl::byte *>(axes_array), 16},
                            true, host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");
    int32_t strides_array[] = {1, 1, 1, 1};
    auto strides =
        hrt::create(input.datatype(), {4},
                    {reinterpret_cast<gsl::byte *>(strides_array), 16}, true,
                    host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    auto output =
        kernels::stackvm::slice(input.impl(), begin.impl(), end.impl(),
                                axes.impl(), strides.impl())
            .expect("slice failed");
    runtime_tensor actual(output.as<tensor>().expect("as tensor failed"));

    // compare
    EXPECT_FALSE(is_same_tensor(expected, actual));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}