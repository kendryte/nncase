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
#include <c_api.h>
#include <gtest/gtest.h>
#include <iostream>
#include <nncase/kernels/stackvm/tensor_ops.h>
#include <nncase/runtime/datatypes.h>
#include <nncase/runtime/runtime_tensor.h>
#include <nncase/runtime/simple_types.h>
#include <nncase/runtime/stackvm/opcode.h>
#include <operators.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace ortki;

class BatchNormalizationTest
    : public KernelTest,
      public ::testing::TestWithParam<std::tuple<
          nncase::typecode_t, dims_t, dims_t, dims_t, dims_t, dims_t>> {
  public:
    void SetUp() override {
        auto &&[typecode, input_shape, scale_shape, b_shape, mean_shape,
                var_shape] = GetParam();

        input = hrt::create(typecode, input_shape,
                            host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");
        init_tensor(input);

        scale = hrt::create(typecode, scale_shape,
                            host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");
        init_tensor(scale);

        b = hrt::create(typecode, b_shape, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
        init_tensor(b);

        mean = hrt::create(typecode, mean_shape,
                           host_runtime_tensor::pool_cpu_only)
                   .expect("create tensor failed");
        init_tensor(mean);

        var =
            hrt::create(typecode, var_shape, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
        init_tensor(var);
    }

    void TearDown() override {}

  protected:
    runtime_tensor input;
    runtime_tensor scale;
    runtime_tensor b;
    runtime_tensor mean;
    runtime_tensor var;
};

INSTANTIATE_TEST_SUITE_P(batch_normalization, BatchNormalizationTest,
                         testing::Combine(testing::Values(dt_float32),
                                          testing::Values(dims_t{1, 3, 16, 16}),
                                          testing::Values(dims_t{3}),
                                          testing::Values(dims_t{3}),
                                          testing::Values(dims_t{3}),
                                          testing::Values(dims_t{3})));

TEST_P(BatchNormalizationTest, batch_normalization) {
    auto input_ort = runtime_tensor_2_ort_tensor(input);
    auto scale_ort = runtime_tensor_2_ort_tensor(scale);
    auto b_ort = runtime_tensor_2_ort_tensor(b);
    auto mean_ort = runtime_tensor_2_ort_tensor(mean);
    auto var_ort = runtime_tensor_2_ort_tensor(var);

    // expected
    auto output_ort = ortki_BatchNormalization(input_ort, scale_ort, b_ort,
                                               mean_ort, var_ort, 0.01f, 0.5f);
    size_t size = 0;
    void *ptr_ort = tensor_buffer(output_ort, &size);
    dims_t shape(tensor_rank(output_ort));
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(nncase::dt_float32, shape,
                                {reinterpret_cast<gsl::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    float *epsilon_ptr;
    *epsilon_ptr = 0.01f;
    auto epsilon =
        hrt::create(nncase::dt_float32, {1},
                    {reinterpret_cast<gsl::byte *>(epsilon_ptr), sizeof(float)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    float *monentum_ptr;
    *monentum_ptr = 0.5f;
    auto monentum =
        hrt::create(nncase::dt_float32, {1},
                    {reinterpret_cast<gsl::byte *>(monentum_ptr), sizeof(float)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    // actual
    auto output = kernels::stackvm::batch_normalization(
                      input.impl(), scale.impl(), b.impl(), mean.impl(),
                      var.impl(), epsilon.impl(), monentum.impl())
                      .expect("batch_normalization failed");
    runtime_tensor actual(output.as<tensor>().expect("as tensor failed"));

    // compare
    EXPECT_TRUE(is_same_tensor(expected, actual));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}