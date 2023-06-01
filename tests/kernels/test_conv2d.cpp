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

class Conv2DTest : public KernelTest,
                   public ::testing::TestWithParam<
                       std::tuple<nncase::typecode_t, dims_t, dims_t, dims_t>> {
  public:
    void SetUp() override {
        auto &&[typecode, input_shape, weight_shape, bias_shape] = GetParam();

        input = hrt::create(typecode, input_shape,
                            host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");
        init_tensor(input);

        weight = hrt::create(typecode, weight_shape,
                             host_runtime_tensor::pool_cpu_only)
                     .expect("create tensor failed");
        init_tensor(weight);

        bais = hrt::create(typecode, bias_shape,
                           host_runtime_tensor::pool_cpu_only)
                   .expect("create tensor failed");
        init_tensor(bais);
    }

    void TearDown() override {}

  protected:
    runtime_tensor input;
    runtime_tensor weight;
    runtime_tensor bais;
};

INSTANTIATE_TEST_SUITE_P(Conv2D, Conv2DTest,
                         testing::Combine(testing::Values(dt_float32),
                                          testing::Values(dims_t{1, 4, 5, 5}),
                                          testing::Values(dims_t{8, 4, 3, 3}),
                                          testing::Values(dims_t{8})));

TEST_P(Conv2DTest, conv2d) {
    auto input_ort = runtime_tensor_2_ort_tensor(input);
    auto weight_ort = runtime_tensor_2_ort_tensor(weight);
    auto bais_ort = runtime_tensor_2_ort_tensor(bais);

    // expected
    const char *auto_pad = "NOTSET";
    int64_t dilations[] = {1, 1};
    int64_t kernel_shape[] = {3, 3};
    int64_t pad[] = {1, 1, 1, 1};
    int64_t strides[] = {1, 1};
    auto output_ort =
        ortki_Conv(input_ort, weight_ort, bais_ort, auto_pad, dilations, 2, 1,
                   kernel_shape, 2, pad, 4, strides, 2);
    size_t size = 0;
    void *ptr_ort = tensor_buffer(output_ort, &size);
    dims_t shape(tensor_rank(output_ort));
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(dt_float32, shape,
                                {reinterpret_cast<gsl::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    // actual
    int64_t group[] = {1};
    float_t fused_clamp[] = {FLT_MIN, FLT_MAX};
    auto dilations_ptr = hrt::create(nncase::dt_float32, {2},
                                     {reinterpret_cast<gsl::byte *>(dilations),
                                      2 * sizeof(float)},
                                     true, host_runtime_tensor::pool_cpu_only)
                             .expect("create tensor failed");
    auto kernel_shape_ptr =
        hrt::create(
            nncase::dt_float32, {2},
            {reinterpret_cast<gsl::byte *>(kernel_shape), 2 * sizeof(float)},
            true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    auto pad_ptr =
        hrt::create(nncase::dt_float32, {4},
                    {reinterpret_cast<gsl::byte *>(pad), 4 * sizeof(float)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    auto strides_ptr =
        hrt::create(nncase::dt_float32, {2},
                    {reinterpret_cast<gsl::byte *>(strides), 2 * sizeof(float)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    auto group_ptr =
        hrt::create(nncase::dt_float32, {1},
                    {reinterpret_cast<gsl::byte *>(group), sizeof(float)}, true,
                    host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    auto fused_clamp_ptr =
        hrt::create(
            nncase::dt_float32, {2},
            {reinterpret_cast<gsl::byte *>(fused_clamp), 2 * sizeof(float)},
            true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    auto output =
        kernels::stackvm::conv2d(
            runtime::stackvm::pad_mode_t::constant, input.impl(), weight.impl(),
            bais.impl(), strides_ptr.impl(), pad_ptr.impl(),
            dilations_ptr.impl(), group_ptr.impl(), fused_clamp_ptr.impl())
            .expect("conv2d failed");
    runtime_tensor actual(output.as<tensor>().expect("as tensor failed"));

    // compare
    EXPECT_TRUE(is_same_tensor(expected, actual));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}