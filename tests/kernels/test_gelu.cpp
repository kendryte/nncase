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

#define TEST_CASE_NAME "test_gelu"

using namespace nncase;
using namespace nncase::runtime;
using namespace ortki;

class GeluTest : public KernelTest,
                 public ::testing::TestWithParam<std::tuple<int>> {
  public:
    void SetUp() override {
        READY_SUBCASE()

        auto l_shape = GetShapeArray("lhs_shape");
        auto typecode = GetDataType("lhs_type");

        input =
            hrt::create(typecode, l_shape, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
        init_tensor(input);

        alpha = hrt::create(typecode, {1}, host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");
        init_tensor_alpha(alpha);
    }

    void TearDown() override { CLEAR_SUBCASE() }

    virtual void init_tensor_alpha(runtime::runtime_tensor &tensor) {
        auto dtype = tensor.datatype();
        switch (dtype) {
        case dt_float16: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dis(0.0f, 2.0f);
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<half>(tensor, index) = static_cast<half>(dis(gen));
                    return ok();
                });
            break;
        }
        case dt_float32: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dis(0.0f, 2.0f);
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<float>(tensor, index) = static_cast<float>(dis(gen));
                    return ok();
                });
            break;
        }
        case dt_float64: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<double> dis(0.0, 2.0);
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<double>(tensor, index) = static_cast<double>(dis(gen));
                    return ok();
                });
            break;
        }
        case dt_bfloat16: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(0.0, 2.0);
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<bfloat16>(tensor, index) =
                        static_cast<bfloat16>(dis(gen));
                    return ok();
                });
            break;
        }
        default: {
        }
        }
    }

  protected:
    runtime_tensor input;
    runtime_tensor alpha;
};

INSTANTIATE_TEST_SUITE_P(gelu, GeluTest,
                         testing::Combine(testing::Range(0, MAX_CASE_NUM)));

TEST_P(GeluTest, gelu) {
    auto l_ort = runtime_tensor_2_ort_tensor(input);

    // expected
    auto a_ort = runtime_tensor_2_ort_tensor(alpha);

    runtime_tensor b;
    runtime_tensor c;
    if (input.datatype() == dt_float16) {
        half b_ptr[] = {(half)2.0f};
        b = hrt::create(nncase::dt_float16, {1},
                        {reinterpret_cast<gsl::byte *>(b_ptr), sizeof(b_ptr)},
                        true, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");

        half c_ptr[] = {(half)1.0f};
        c = hrt::create(nncase::dt_float16, {1},
                        {reinterpret_cast<gsl::byte *>(c_ptr), sizeof(c_ptr)},
                        true, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
    } else if (input.datatype() == dt_float32) {
        float b_ptr[] = {2.0f};
        b = hrt::create(nncase::dt_float32, {1},
                        {reinterpret_cast<gsl::byte *>(b_ptr), sizeof(b_ptr)},
                        true, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");

        float c_ptr[] = {1.0f};
        c = hrt::create(nncase::dt_float32, {1},
                        {reinterpret_cast<gsl::byte *>(c_ptr), sizeof(c_ptr)},
                        true, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
    } else if (input.datatype() == dt_float64) {
        double b_ptr[] = {2.0f};
        b = hrt::create(nncase::dt_float64, {1},
                        {reinterpret_cast<gsl::byte *>(b_ptr), sizeof(b_ptr)},
                        true, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");

        double c_ptr[] = {1.0f};
        c = hrt::create(nncase::dt_float64, {1},
                        {reinterpret_cast<gsl::byte *>(c_ptr), sizeof(c_ptr)},
                        true, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
    }

    auto b_ort = runtime_tensor_2_ort_tensor(b);
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
    auto output = kernels::stackvm::gelu(input.impl(), alpha.impl())
                      .expect("gelu failed");
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
    FOR_LOOP(lhs_shape, j)
    FOR_LOOP(lhs_type, i)
    SPLIT_ELEMENT(lhs_shape, j)
    SPLIT_ELEMENT(lhs_type, i)
    WRITE_SUB_CASE()
    FOR_LOOP_END()
    FOR_LOOP_END()

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}