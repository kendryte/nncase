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

class QuantizeTest : public KernelTest,
                     public ::testing::TestWithParam<
                         std::tuple<nncase::typecode_t, typecode_t, dims_t>> {
  public:
    void SetUp() override {
        auto &&[input_typecode, quant_type, l_shape] = GetParam();

        input = hrt::create(input_typecode, l_shape,
                            host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");
        init_tensor(input);

        auto zero_point_value = 127;
        auto scale_value = 0.01f;

        if (quant_type == dt_uint8) {
            uint8_t zero_point[] = {(uint8_t)zero_point_value};
            zero_point_ptr =
                hrt::create(nncase::dt_uint8, {1},
                            {reinterpret_cast<gsl::byte *>(zero_point),
                             sizeof(zero_point)},
                            true, host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");
        } else if (quant_type == dt_int8) {
            int8_t zero_point[] = {(int8_t)zero_point_value};
            zero_point_ptr =
                hrt::create(nncase::dt_int8, {1},
                            {reinterpret_cast<gsl::byte *>(zero_point),
                             sizeof(zero_point)},
                            true, host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");
        } else {
            int16_t zero_point[] = {(int16_t)zero_point_value};
            zero_point_ptr =
                hrt::create(nncase::dt_int16, {1},
                            {reinterpret_cast<gsl::byte *>(zero_point),
                             sizeof(zero_point)},
                            true, host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");
        }

        float_t scale[] = {scale_value};
        scale_ptr =
            hrt::create(nncase::dt_float32, {1},
                        {reinterpret_cast<gsl::byte *>(scale), sizeof(scale)},
                        true, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");

        quant_param_t quantParam;
        quantParam.zero_point = zero_point_value;
        quantParam.scale = scale_value;
        quant_param_t quant_param[] = {quantParam};
        quant_param_ptr =
            hrt::create(dt_int64, {1},
                        {reinterpret_cast<gsl::byte *>(quant_param),
                         sizeof(quant_param)},
                        true, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
    }

    void TearDown() override {}

  protected:
    runtime_tensor input;
    runtime_tensor scale_ptr;
    runtime_tensor zero_point_ptr;
    runtime_tensor quant_param_ptr;
};

INSTANTIATE_TEST_SUITE_P(
    quantize, QuantizeTest,
    testing::Combine(testing::Values(dt_float32),
                     testing::Values(dt_uint8, dt_int8, dt_int16),
                     testing::Values(dims_t{2, 4}, dims_t{1, 3, 16, 16},
                                     dims_t{1, 3, 16}, dims_t{1, 3}, dims_t{1},
                                     dims_t{}, dims_t{16, 16})));

TEST_P(QuantizeTest, quantize) {
    auto l_ort = runtime_tensor_2_ort_tensor(input);

    if (zero_point_ptr.datatype() != dt_int16) {

        // expected
        runtime_tensor expected;
        auto output_ort = ortki_QuantizeLinear(
            l_ort, runtime_tensor_2_ort_tensor(scale_ptr),
            runtime_tensor_2_ort_tensor(zero_point_ptr), 0);
        size_t size = 0;
        void *ptr_ort = tensor_buffer(output_ort, &size);
        dims_t shape(tensor_rank(output_ort));
        tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
        expected = hrt::create(zero_point_ptr.datatype(), shape,
                               {reinterpret_cast<gsl::byte *>(ptr_ort), size},
                               true, host_runtime_tensor::pool_cpu_only)
                       .expect("create tensor failed");

        // actual
        auto output =
            kernels::stackvm::quantize(zero_point_ptr.datatype(), input.impl(),
                                       quant_param_ptr.impl())
                .expect("quantize failed");
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

    } else {

        // expected
        runtime_tensor expected;
        expected = hrt::create(dt_int16, input.shape(),
                               host_runtime_tensor::pool_cpu_only)
                       .expect("create tensor failed");
        quantize_to_int16(expected, input, 127, 0.01f);

        // actual
        auto output =
            kernels::stackvm::quantize(zero_point_ptr.datatype(), input.impl(),
                                       quant_param_ptr.impl())
                .expect("quantize failed");
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
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}