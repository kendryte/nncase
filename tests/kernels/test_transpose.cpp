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

class TransposeTest
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

INSTANTIATE_TEST_SUITE_P(
    Transpose, TransposeTest,
    testing::Combine(testing::Values(dt_float32, dt_int32, dt_int16, dt_int8,
                                     dt_uint8),
                     testing::Values(dims_t{1, 3, 16, 16}, dims_t{1, 2, 4, 8},
                                     dims_t{2, 2, 4, 4})));

TEST_P(TransposeTest, Transpose) {
    auto input_ort = runtime_tensor_2_ort_tensor(input);
    int64_t perm[] = {1, 0, 3, 2};
    size_t perm_size = 4;

    // expected
    auto output_ort = ortki_Transpose(input_ort, perm, perm_size);
    size_t size = 0;
    void *ptr_ort = tensor_buffer(output_ort, &size);
    dims_t shape(tensor_rank(output_ort));
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(input.datatype(), shape,
                                {reinterpret_cast<gsl::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    auto perm1 =
        hrt::create(nncase::dt_int64, {4},
                    {reinterpret_cast<gsl::byte *>(perm), sizeof(perm)}, true,
                    host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    int32_t perm_size_ptr[] = {4};
    auto perm_size1 = hrt::create(nncase::dt_int32, {1},
                                  {reinterpret_cast<gsl::byte *>(perm_size_ptr),
                                   sizeof(perm_size_ptr)},
                                  true, host_runtime_tensor::pool_cpu_only)
                          .expect("create tensor failed");

    auto output = kernels::stackvm::transpose(input.impl(), perm1.impl())
                      .expect("transpose failed");
    runtime_tensor actual(output.as<tensor>().expect("as tensor failed"));

    // compare
    EXPECT_TRUE(is_same_tensor(expected, actual));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}