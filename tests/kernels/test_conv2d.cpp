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
                       std::tuple<nncase::typecode_t, dims_t, dims_t, dims_t,
                                  dims_t, dims_t, dims_t, int64_t>> {
  public:
    void SetUp() override {
        auto &&[typecode, input_shape, weight_shape, bias_shape, value1, value2,
                value3, value4] = GetParam();

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

        dilations_value = value1;
        pad_value = value2;
        strides_value = value3;
        group_value = value4;
    }

    void TearDown() override {}

  protected:
    runtime_tensor input;
    runtime_tensor weight;
    runtime_tensor bais;
    dims_t dilations_value;
    dims_t pad_value;
    dims_t strides_value;
    int64_t group_value;
};

INSTANTIATE_TEST_SUITE_P(
    conv2d, Conv2DTest,
    testing::Combine(
        testing::Values(dt_float32),
        testing::Values(dims_t{1, 4, 5, 5}, dims_t{1, 4, 16, 16}),
        testing::Values(dims_t{8, 4, 3, 3}, dims_t{8, 4, 1, 1}),
        testing::Values(dims_t{8}), testing::Values(dims_t{2, 2}, dims_t{1, 1}),
        testing::Values(dims_t{1, 1, 1, 1} /*, dims_t{0, 0, 1, 0}*/),
        testing::Values(dims_t{1, 1}, dims_t{2, 2}),
        testing::Values(1 /*, 2*/))); // todo result error

TEST_P(Conv2DTest, conv2d) {
    auto input_ort = runtime_tensor_2_ort_tensor(input);
    auto weight_ort = runtime_tensor_2_ort_tensor(weight);
    auto bais_ort = runtime_tensor_2_ort_tensor(bais);

    // expected
    const char auto_pad[7] = "NOTSET";

    size_t dilations_size = dilations_value.size();
    int64_t *dilations = (int64_t *)malloc(dilations_size * sizeof(int64_t));
    std::copy(dilations_value.begin(), dilations_value.end(), dilations);

    int64_t kernel_shape[] = {(int64_t)weight.shape()[2],
                              (int64_t)weight.shape()[3]};

    size_t pad_size = pad_value.size();
    int64_t *pad = (int64_t *)malloc(pad_size * sizeof(int64_t));
    std::copy(pad_value.begin(), pad_value.end(), pad);

    size_t strides_size = strides_value.size();
    int64_t *strides = (int64_t *)malloc(strides_size * sizeof(int64_t));
    std::copy(strides_value.begin(), strides_value.end(), strides);

    auto output_ort = ortki_Conv(
        input_ort, weight_ort, bais_ort, auto_pad, dilations, dilations_size,
        group_value, kernel_shape, 2, pad, pad_size, strides, strides_size);
    size_t size = 0;
    void *ptr_ort = tensor_buffer(output_ort, &size);
    dims_t shape(tensor_rank(output_ort));
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(dt_float32, shape,
                                {reinterpret_cast<gsl::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    // actual
    int64_t group[] = {group_value};

    float fused_clamp[] = {-std::numeric_limits<float>::infinity(),
                           std::numeric_limits<float>::infinity()};

    auto dilations_ptr = hrt::create(nncase::dt_int64, {2},
                                     {reinterpret_cast<gsl::byte *>(dilations),
                                      dilations_size * sizeof(int64_t)},
                                     true, host_runtime_tensor::pool_cpu_only)
                             .expect("create tensor failed");

    auto kernel_shape_ptr =
        hrt::create(
            nncase::dt_int64, {2},
            {reinterpret_cast<gsl::byte *>(kernel_shape), sizeof(kernel_shape)},
            true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    auto pad_ptr = hrt::create(nncase::dt_int64, {4},
                               {reinterpret_cast<gsl::byte *>(pad),
                                pad_size * sizeof(int64_t)},
                               true, host_runtime_tensor::pool_cpu_only)
                       .expect("create tensor failed");

    auto strides_ptr = hrt::create(nncase::dt_int64, {2},
                                   {reinterpret_cast<gsl::byte *>(strides),
                                    strides_size * sizeof(int64_t)},
                                   true, host_runtime_tensor::pool_cpu_only)
                           .expect("create tensor failed");

    auto group_ptr =
        hrt::create(nncase::dt_int64, {1},
                    {reinterpret_cast<gsl::byte *>(group), sizeof(group)}, true,
                    host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    auto fused_clamp_ptr =
        hrt::create(
            nncase::dt_float32, {2},
            {reinterpret_cast<gsl::byte *>(fused_clamp), sizeof(fused_clamp)},
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
    bool result = is_same_tensor(expected, actual) ||
                  cosine_similarity_tensor(expected, actual);

    if (!result) {
        std::cout << "actual ";
        print_runtime_tensor(actual);
        std::cout << "expected ";
        print_runtime_tensor(expected);
    }

    EXPECT_TRUE(result);
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}