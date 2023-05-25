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

class GatherTest
    : public KernelTest,
      public ::testing::TestWithParam<std::tuple<nncase::typecode_t, dims_t>> {
  public:
    void SetUp() override {
        auto &&[typecode, shape] = GetParam();

        size_t size = 0;
        int input_array[] = {0, 1, 2, 3};
        input = hrt::create(dt_int32, shape,
                            {reinterpret_cast<gsl::byte *>(input_array), size},
                            true, host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");

        long indices_array[] = {0, 0, 1, 1};
        indices =
            hrt::create(dt_int64, shape,
                        {reinterpret_cast<gsl::byte *>(indices_array), size},
                        true, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");

        long batchDims_array[] = {0};
        batchDims =
            hrt::create(dt_int64, shape,
                        {reinterpret_cast<gsl::byte *>(batchDims_array), size},
                        true, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
    }

    void TearDown() override {}

  protected:
    runtime_tensor input;
    runtime_tensor indices;
    runtime_tensor batchDims;
};

INSTANTIATE_TEST_SUITE_P(Gather, GatherTest,
                         testing::Combine(testing::Values(dt_int32, dt_int64),
                                          testing::Values(dims_t{2, 2})));

TEST_P(GatherTest, gather) {
    auto input_ort = runtime_tensor_2_ort_tensor(input);
    auto indices_ort = runtime_tensor_2_ort_tensor(indices);

    // expected
    auto output_ort = ortki_Gather(input_ort, indices_ort, 0);
    size_t size = 0;
    void *ptr_ort = tensor_buffer(output_ort, &size);
    dims_t shape(tensor_rank(output_ort));
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(input.datatype(), shape,
                                {reinterpret_cast<gsl::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    // actual
    auto output =
        kernels::stackvm::gather(input.impl(), batchDims.impl(), indices.impl())
            .expect("gather failed");
    runtime_tensor actual(output.as<tensor>().expect("as tensor failed"));

    // compare
    EXPECT_TRUE(is_same_tensor(expected, actual));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}