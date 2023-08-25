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

#define TEST_CASE_NAME "test_hard_sigmoid"

using namespace nncase;
using namespace nncase::runtime;
using namespace ortki;

class HardSigmoidTest : public KernelTest,
                        public ::testing::TestWithParam<std::tuple<int>> {
  public:
    void SetUp() override {
        READY_SUBCASE()

        auto l_shape = GetShapeArray("lhs_shape");
        auto typecode = GetDataType("lhs_type");
        auto value1 = GetFloatNumber("alpha");
        auto value2 = GetFloatNumber("gamma");

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

INSTANTIATE_TEST_SUITE_P(hard_sigmoid, HardSigmoidTest,
                         testing::Combine(testing::Range(0, MAX_CASE_NUM)));

TEST_P(HardSigmoidTest, hard_sigmoid) {
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

    auto output_ort = ortki_HardSigmoid(l_ort, alpha_value, gamma_value);
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
        kernels::stackvm::hard_sigmoid(input.impl(), alpha.impl(), gamma.impl())
            .expect("hard_sigmoid failed");
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
    READY_TEST_CASE_GENERATE()
    FOR_LOOP(lhs_shape, i)
    FOR_LOOP(lhs_type, j)
    FOR_LOOP(alpha, k)
    FOR_LOOP(gamma, l)
    SPLIT_ELEMENT(lhs_shape, i)
    SPLIT_ELEMENT(lhs_type, j)
    SPLIT_ELEMENT(alpha, k)
    SPLIT_ELEMENT(gamma, l)
    WRITE_SUB_CASE()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}