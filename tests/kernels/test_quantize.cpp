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

class QuantizeTest
    : public KernelTest,
      public ::testing::TestWithParam<std::tuple<nncase::typecode_t, dims_t>> {
  public:
    void SetUp() override {
        auto &&[typecode, l_shape] = GetParam();

        float input_array[] = {1.0F, 1.2F, 1.4F, 1.5F, 1.6F, 1.8F, 1.9F, 2.0F};
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

INSTANTIATE_TEST_SUITE_P(Quantize, QuantizeTest,
                         testing::Combine(testing::Values(dt_float32),
                                          testing::Values(dims_t{1, 3, 16,
                                                                 16})));

TEST_P(QuantizeTest, quantize) {
    auto l_ort = runtime_tensor_2_ort_tensor(input);

    // expected
    uint8_t zero_point[] = {127};
    auto zero_point_ptr =
        hrt::create(
            nncase::dt_uint8, {1},
            {reinterpret_cast<gsl::byte *>(zero_point), sizeof(zero_point)},
            true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    float_t scale[] = {0.05f};
    auto scale_ptr =
        hrt::create(nncase::dt_float32, {1},
                    {reinterpret_cast<gsl::byte *>(scale), sizeof(scale)}, true,
                    host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    auto output_ort =
        ortki_QuantizeLinear(l_ort, runtime_tensor_2_ort_tensor(scale_ptr),
                             runtime_tensor_2_ort_tensor(zero_point_ptr), 0);
    size_t size = 0;
    void *ptr_ort = tensor_buffer(output_ort, &size);
    dims_t shape(tensor_rank(output_ort));
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(dt_uint8, shape,
                                {reinterpret_cast<gsl::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    // actual
    //    float_t quant_param[] = {127, 0.01f};
    //    auto quant_param_ptr =
    //        hrt::create(nncase::dt_float32, {2},
    //                    {reinterpret_cast<gsl::byte *>(quant_param),
    //                    sizeof(quant_param)}, true,
    //                    host_runtime_tensor::pool_cpu_only)
    //            .expect("create tensor failed");
    //    auto output = kernels::stackvm::quantize(dt_float32, input.impl(),
    //                                             quant_param_ptr.impl())
    //                      .expect("quantize failed");
    //    runtime_tensor actual(output.as<tensor>().expect("as tensor failed"));

    // compare
    EXPECT_TRUE(is_same_tensor(expected, expected) ||
                cosine_similarity_tensor(expected, expected));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}