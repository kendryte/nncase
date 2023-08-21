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

class GeluTest : public KernelTest,
                 public ::testing::TestWithParam<
                     std::tuple<nncase::typecode_t, dims_t, float>> {
  public:
    void SetUp() override {
        auto &&[typecode, l_shape, a_value] = GetParam();

        input =
            hrt::create(typecode, l_shape, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
        init_tensor(input);

        a = a_value;
    }

    void TearDown() override {}

  protected:
    runtime_tensor input;
    float a;
};

INSTANTIATE_TEST_SUITE_P(
    gelu, GeluTest,
    testing::Combine(testing::Values(dt_float32),
                     testing::Values(dims_t{1, 3, 16, 16}, dims_t{1},
                                     dims_t{8, 8}, dims_t{1, 4, 16},
                                     dims_t{1, 3, 24, 24}, dims_t{}),
                     testing::Values(1.2f, 0.8f, 0.5f)));

TEST_P(GeluTest, gelu) {
    auto l_ort = runtime_tensor_2_ort_tensor(input);

    // expected
    float a_ptr[] = {a};
    auto a = hrt::create(nncase::dt_float32, {1},
                         {reinterpret_cast<gsl::byte *>(a_ptr), sizeof(a_ptr)},
                         true, host_runtime_tensor::pool_cpu_only)
                 .expect("create tensor failed");
    auto a_ort = runtime_tensor_2_ort_tensor(a);

    float b_ptr[] = {2.0f};
    auto b = hrt::create(nncase::dt_float32, {1},
                         {reinterpret_cast<gsl::byte *>(b_ptr), sizeof(b_ptr)},
                         true, host_runtime_tensor::pool_cpu_only)
                 .expect("create tensor failed");
    auto b_ort = runtime_tensor_2_ort_tensor(b);

    float c_ptr[] = {1.0f};
    auto c = hrt::create(nncase::dt_float32, {1},
                         {reinterpret_cast<gsl::byte *>(c_ptr), sizeof(c_ptr)},
                         true, host_runtime_tensor::pool_cpu_only)
                 .expect("create tensor failed");
    auto c_ort = runtime_tensor_2_ort_tensor(c);

    auto scaledInput = ortki_Mul(a_ort, l_ort);
    auto output_ort = ortki_Mul(
        a_ort,
        ortki_Mul(scaledInput, ortki_Add(ortki_Erf(ortki_Div(
                                             scaledInput, ortki_Sqrt(b_ort))),
                                         c_ort)));
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
        kernels::stackvm::gelu(input.impl(), a.impl()).expect("gelu failed");
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