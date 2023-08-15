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

#define TEST_CASE_NAME "test_batch_normalization"

using namespace nncase;
using namespace nncase::runtime;
using namespace ortki;

class BatchNormalizationTest
    : public KernelTest,
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

        scale = hrt::create(typecode, {input_shape[1]},
                            host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");
        init_tensor(scale);

        b = hrt::create(typecode, {input_shape[1]},
                        host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
        init_tensor(b);

        mean = hrt::create(typecode, {input_shape[1]},
                           host_runtime_tensor::pool_cpu_only)
                   .expect("create tensor failed");
        init_tensor(mean);

        var = hrt::create(typecode, {input_shape[1]},
                          host_runtime_tensor::pool_cpu_only)
                  .expect("create tensor failed");
        init_tensor_var(var);
    }

    void TearDown() override {}

    virtual void init_tensor_var(runtime::runtime_tensor &tensor) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.1f, 6.0f);
        NNCASE_UNUSED auto res = kernels::stackvm::apply(
            tensor.shape(), [&](const dims_t &index) -> result<void> {
                get<float>(tensor, index) = static_cast<float>(dis(gen));
                return ok();
            });
    }

  protected:
    runtime_tensor input;
    runtime_tensor scale;
    runtime_tensor b;
    runtime_tensor mean;
    runtime_tensor var;
};

INSTANTIATE_TEST_SUITE_P(batch_normalization, BatchNormalizationTest,
                         testing::Combine(testing::Range(0, MAX_CASE_NUM)));

TEST_P(BatchNormalizationTest, batch_normalization) {
    auto input_ort = runtime_tensor_2_ort_tensor(input);
    auto scale_ort = runtime_tensor_2_ort_tensor(scale);
    auto b_ort = runtime_tensor_2_ort_tensor(b);
    auto mean_ort = runtime_tensor_2_ort_tensor(mean);
    auto var_ort = runtime_tensor_2_ort_tensor(var);

    auto eps = 0.01f;
    auto momentum = 0.9f;

    // expected
    auto output_ort = ortki_BatchNormalization(
        input_ort, scale_ort, b_ort, mean_ort, var_ort, eps, momentum);
    size_t size = 0;
    void *ptr_ort = tensor_buffer(output_ort, &size);
    dims_t shape(tensor_rank(output_ort));
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(input.datatype(), shape,
                                {reinterpret_cast<gsl::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    float epsilon_ptr[] = {eps};
    auto epsilon = hrt::create(nncase::dt_float32, {1},
                               {reinterpret_cast<gsl::byte *>(epsilon_ptr),
                                sizeof(epsilon_ptr)},
                               true, host_runtime_tensor::pool_cpu_only)
                       .expect("create tensor failed");

    float monentum_ptr[] = {momentum};
    auto monentum = hrt::create(nncase::dt_float32, {1},
                                {reinterpret_cast<gsl::byte *>(monentum_ptr),
                                 sizeof(monentum_ptr)},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    // actual
    auto output = kernels::stackvm::batch_normalization(
                      input.impl(), scale.impl(), b.impl(), mean.impl(),
                      var.impl(), epsilon.impl(), monentum.impl())
                      .expect("batch_normalization failed");
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