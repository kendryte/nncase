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

class GatherTest : public KernelTest,
                   public ::testing::TestWithParam<
                       std::tuple<nncase::typecode_t, dims_t, int64_t>> {
  public:
    void SetUp() override {
        auto &&[typecode, shape, value] = GetParam();

        input = hrt::create(typecode, shape, host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");
        init_tensor(input);

        int64_t indices_array[] = {0, 0, -1, -1};
        indices = hrt::create(dt_int64, {4},
                              {reinterpret_cast<gsl::byte *>(indices_array),
                               sizeof(indices_array)},
                              true, host_runtime_tensor::pool_cpu_only)
                      .expect("create tensor failed");

        batchDims_value = value >= 0
                              ? (size_t)value >= shape.size() ? -1 : value
                          : -(size_t)value > shape.size() ? -1
                                                          : value;

        int64_t batchDims_array[1] = {batchDims_value};
        batchDims = hrt::create(dt_int64, dims_t{1},
                                {reinterpret_cast<gsl::byte *>(batchDims_array),
                                 sizeof(batchDims_array)},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");
    }

    void TearDown() override {}

  protected:
    runtime_tensor input;
    runtime_tensor indices;
    runtime_tensor batchDims;
    int64_t batchDims_value;
};

INSTANTIATE_TEST_SUITE_P(
    gather, GatherTest,
    testing::Combine(testing::Values(dt_int32, dt_int64, dt_float32, dt_uint64,
                                     dt_int8, dt_int16, dt_uint8, dt_uint16,
                                     dt_uint32, dt_float16, dt_float64,
                                     dt_bfloat16, dt_boolean),
                     testing::Values(dims_t{2, 3, 5, 7}, dims_t{2, 2},
                                     dims_t{2, 3, 1}, dims_t{5, 5, 7, 7},
                                     dims_t{11}),
                     testing::Values(-1, 0, 1, -2, -3, 2, 3, -4)));

TEST_P(GatherTest, gather) {
    auto input_ort = runtime_tensor_2_ort_tensor(input);
    auto indices_ort = runtime_tensor_2_ort_tensor(indices);

    // expected
    auto output_ort = ortki_Gather(input_ort, indices_ort, batchDims_value);
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