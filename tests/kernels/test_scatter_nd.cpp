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

class ScatterNDTest
    : public KernelTest,
      public ::testing::TestWithParam<
          std::tuple<nncase::typecode_t, typecode_t, dims_t, dims_t, dims_t>> {
  public:
    void SetUp() override {
        auto &&[typecode1, typecode2, input_shape, indices_shape,
                updates_shape] = GetParam();

        input = hrt::create(typecode1, input_shape,
                            host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");
        init_tensor(input);

        int64_t indices_array[] = {0, 0, 1, 1, 0, 1};
        indices = hrt::create(typecode2, indices_shape,
                              {reinterpret_cast<gsl::byte *>(indices_array),
                               sizeof(indices_array)},
                              true, host_runtime_tensor::pool_cpu_only)
                      .expect("create tensor failed");

        updates = hrt::create(typecode1, updates_shape,
                              host_runtime_tensor::pool_cpu_only)
                      .expect("create tensor failed");
        init_tensor(updates);
    }

    void TearDown() override {}

  protected:
    runtime_tensor input;
    runtime_tensor indices;
    runtime_tensor updates;
};

INSTANTIATE_TEST_SUITE_P(
    ScatterND, ScatterNDTest,
    testing::Combine(testing::Values(dt_float32, dt_uint8, dt_int8, dt_float16,
                                     dt_uint32, dt_uint64, dt_uint16, dt_int16,
                                     dt_int32, dt_int64, dt_float64, dt_boolean,
                                     dt_bfloat16),
                     testing::Values(dt_int64),
                     testing::Values(dims_t{2, 1, 10}, dims_t{2, 5, 10}),
                     testing::Values(dims_t{2, 1, 1, 3}),
                     testing::Values(dims_t{2, 1, 1})));

TEST_P(ScatterNDTest, ScatterND) {

    // expected
    auto input_ort = runtime_tensor_2_ort_tensor(input);
    auto indices_ort = runtime_tensor_2_ort_tensor(indices);
    auto updates_ort = runtime_tensor_2_ort_tensor(updates);
    auto output_ort =
        ortki_ScatterND(input_ort, indices_ort, updates_ort, "none");
    size_t size = 0;
    void *ptr_ort = tensor_buffer(output_ort, &size);
    dims_t shape(tensor_rank(output_ort));
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(input.datatype(), shape,
                                {reinterpret_cast<gsl::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    // actual
    auto output = kernels::stackvm::scatter_nd(input.impl(), indices.impl(),
                                               updates.impl())
                      .expect("scatter_nd failed");
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