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

class ClampTest
    : public KernelTest,
      public ::testing::TestWithParam<
          std::tuple<nncase::typecode_t, dims_t, float_t, float_t>> {
  public:
    void SetUp() override {
        auto &&[typecode, l_shape, value1, value2] = GetParam();

        input =
            hrt::create(typecode, l_shape, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
        init_tensor(input);

        min_value = value1;
        max_value = value2;
    }

    void TearDown() override {}

  protected:
    runtime_tensor input;
    float_t min_value;
    float_t max_value;
};

INSTANTIATE_TEST_SUITE_P(
    clamp, ClampTest,
    testing::Combine(testing::Values(dt_float16),
                     testing::Values(dims_t{1, 3, 16, 16}, dims_t{1},
                                     dims_t{1, 3}, dims_t{8, 8},
                                     dims_t{1, 3, 8}, dims_t{16, 16}, dims_t{},
                                     dims_t{16}),
                     testing::Values(-1, -2, -3, -4, -5, -6),
                     testing::Values(1, 2, 3, 4, 5, 6)));

TEST_P(ClampTest, clamp) {

    // expected
    half min1[] = {half(min_value)};
    auto min_tensor =
        hrt::create(nncase::dt_float16, {1},
                    {reinterpret_cast<gsl::byte *>(min1), sizeof(min1)}, true,
                    host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    half max1[] = {half(max_value)};
    auto max_tensor =
        hrt::create(nncase::dt_float16, {1},
                    {reinterpret_cast<gsl::byte *>(max1), sizeof(max1)}, true,
                    host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    auto output_ort =
        ortki_Clip(runtime_tensor_2_ort_tensor(input),
                   ortki_CastLike(runtime_tensor_2_ort_tensor(min_tensor),
                                  runtime_tensor_2_ort_tensor(input)),
                   ortki_CastLike(runtime_tensor_2_ort_tensor(max_tensor),
                                  runtime_tensor_2_ort_tensor(input)));
    size_t size = 0;
    void *ptr_ort = tensor_buffer(output_ort, &size);
    dims_t shape(tensor_rank(output_ort));
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(input.datatype(), shape,
                                {reinterpret_cast<gsl::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    // actual
    auto output = kernels::stackvm::clamp(input.impl(), min_tensor.impl(),
                                          max_tensor.impl())
                      .expect("clamp failed");
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