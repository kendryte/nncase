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

class SeluTest : public KernelTest,
                 public ::testing::TestWithParam<
                     std::tuple<nncase::typecode_t, dims_t, float, float>> {
  public:
    void SetUp() override {
        auto &&[typecode, l_shape, value1, value2] = GetParam();

        input =
            hrt::create(typecode, l_shape, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
        init_tensor(input);

        alpha_value = value1;
        gamma_value = value2;
    }

    void TearDown() override {}

  protected:
    runtime_tensor input;
    float alpha_value;
    float gamma_value;
};

INSTANTIATE_TEST_SUITE_P(
    Selu, SeluTest,
    testing::Combine(testing::Values(dt_float32, dt_float16, dt_float64),
                     testing::Values(dims_t{1, 3, 16, 16}, dims_t{1, 3, 16},
                                     dims_t{1, 3}, dims_t{1}, dims_t{8, 8},
                                     dims_t{}),
                     testing::Values(1.2f, 0.8f, 0.5f, 0.6f, 1.5f),
                     testing::Values(1.2f, 0.8f, 0.5f, 0.6f, 1.5f)));

TEST_P(SeluTest, Selu) {
    auto l_ort = runtime_tensor_2_ort_tensor(input);

    // expected
    runtime_tensor alpha;
    runtime_tensor gamma;
    if (input.datatype() == dt_float32) {
        float alpha_ptr[] = {alpha_value};
        alpha = hrt::create(nncase::dt_float32, {1},
                            {reinterpret_cast<gsl::byte *>(alpha_ptr),
                             sizeof(alpha_ptr)},
                            true, host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");

        float gamma_ptr[] = {gamma_value};
        gamma = hrt::create(nncase::dt_float32, {1},
                            {reinterpret_cast<gsl::byte *>(gamma_ptr),
                             sizeof(gamma_ptr)},
                            true, host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");
    } else if (input.datatype() == dt_float16) {
        half alpha_ptr[] = {(half)alpha_value};
        alpha = hrt::create(nncase::dt_float16, {1},
                            {reinterpret_cast<gsl::byte *>(alpha_ptr),
                             sizeof(alpha_ptr)},
                            true, host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");

        half gamma_ptr[] = {(half)gamma_value};
        gamma = hrt::create(nncase::dt_float16, {1},
                            {reinterpret_cast<gsl::byte *>(gamma_ptr),
                             sizeof(gamma_ptr)},
                            true, host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");
    } else {
        double alpha_ptr[] = {(double)alpha_value};
        alpha = hrt::create(nncase::dt_float64, {1},
                            {reinterpret_cast<gsl::byte *>(alpha_ptr),
                             sizeof(alpha_ptr)},
                            true, host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");

        double gamma_ptr[] = {(double)gamma_value};
        gamma = hrt::create(nncase::dt_float64, {1},
                            {reinterpret_cast<gsl::byte *>(gamma_ptr),
                             sizeof(gamma_ptr)},
                            true, host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");
    }
    auto output_ort = ortki_Selu(l_ort, alpha_value, gamma_value);
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
        kernels::stackvm::selu(input.impl(), alpha.impl(), gamma.impl())
            .expect("selu failed");
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