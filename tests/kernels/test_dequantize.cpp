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

class DequantizeTest
    : public KernelTest,
      public ::testing::TestWithParam<std::tuple<nncase::typecode_t, dims_t>> {
  public:
    void SetUp() override {
        auto &&[typecode, l_shape] = GetParam();

        uint8_t input_array[] = {127, 128, 150, 160, 170, 180, 200, 205};
        input = hrt::create(typecode, {2, 4},
                            {reinterpret_cast<gsl::byte *>(input_array),
                             sizeof(input_array)},
                            true, host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");
    }

    void TearDown() override {}

  protected:
    runtime_tensor input;
};

INSTANTIATE_TEST_SUITE_P(dequantize, DequantizeTest,
                         testing::Combine(testing::Values(dt_uint8),
                                          testing::Values(dims_t{1, 3, 16,
                                                                 16})));

TEST_P(DequantizeTest, dequantize) {
    auto l_ort = runtime_tensor_2_ort_tensor(input);

    // expected
    uint8_t zero_point[] = {127};
    auto zero_point_ptr =
        hrt::create(
            nncase::dt_uint8, {1},
            {reinterpret_cast<gsl::byte *>(zero_point), sizeof(zero_point)},
            true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    float_t scale[] = {0.01f};
    auto scale_ptr =
        hrt::create(nncase::dt_float32, {1},
                    {reinterpret_cast<gsl::byte *>(scale), sizeof(scale)}, true,
                    host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    auto output_ort =
        ortki_DequantizeLinear(l_ort, runtime_tensor_2_ort_tensor(scale_ptr),
                               runtime_tensor_2_ort_tensor(zero_point_ptr), 0);
    size_t size = 0;
    void *ptr_ort = tensor_buffer(output_ort, &size);
    dims_t shape(tensor_rank(output_ort));
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(dt_float32, shape,
                                {reinterpret_cast<gsl::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    // actual
    quant_param_t quantParam;
    quantParam.zero_point = 127;
    quantParam.scale = 0.01f;
    quant_param_t quant_param[] = {quantParam};
    auto quant_param_ptr =
        hrt::create(
            dt_int64, {1},
            {reinterpret_cast<gsl::byte *>(quant_param), sizeof(quant_param)},
            true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    auto output = kernels::stackvm::dequantize(dt_float32, input.impl(),
                                               quant_param_ptr.impl())
                      .expect("dequantize failed");
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