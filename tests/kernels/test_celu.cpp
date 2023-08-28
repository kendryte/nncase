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

#define TEST_CASE_NAME "test_celu"

using namespace nncase;
using namespace nncase::runtime;
using namespace ortki;

class CeluTest : public KernelTest,
                 public ::testing::TestWithParam<std::tuple<int>> {
  public:
    void SetUp() override {
        READY_SUBCASE()

        auto input_shape = GetShapeArray("lhs_shape");
        auto typecode = GetDataType("lhs_type");

        input = hrt::create(typecode, input_shape,
                            host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");
        init_tensor(input);

        alpha = hrt::create(typecode, {1}, host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");
        init_tensor_alpha(alpha);
    }

    void TearDown() override {}

    virtual void init_tensor_alpha(runtime::runtime_tensor &tensor) {
        auto dtype = tensor.datatype();
        switch (dtype) {
        case dt_float16: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dis(-5.0f, 5.0f);
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
            std::uniform_real_distribution<float> dis(-5.0f, 5.0f);
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<float>(tensor, index) = static_cast<float>(dis(gen));
                    return ok();
                });
            break;
        }
        case dt_bfloat16: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(-5.0f, 5.0);
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

INSTANTIATE_TEST_SUITE_P(celu, CeluTest,
                         testing::Combine(testing::Range(0, MAX_CASE_NUM)));

TEST_P(CeluTest, celu) {
    auto input_ort = runtime_tensor_2_ort_tensor(input);

    // expected
    OrtKITensor *output_ort;
    output_ort = ortki_Celu(input_ort, tensor_to_array<float>(alpha)[0]);
    size_t size = 0;
    void *ptr_ort = tensor_buffer(output_ort, &size);
    dims_t shape(tensor_rank(output_ort));
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(input.datatype(), shape,
                                {reinterpret_cast<gsl::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    // actual
    auto output = kernels::stackvm::celu(input.impl(), alpha.impl())
                      .expect("celu failed");
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
    SPLIT_ELEMENT(lhs_shape, i)
    SPLIT_ELEMENT(lhs_type, j)
    WRITE_SUB_CASE()
    FOR_LOOP_END()
    FOR_LOOP_END()

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}