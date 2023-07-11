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

class NormalTest : public KernelTest,
                   public ::testing::TestWithParam<
                       std::tuple<nncase::typecode_t, dims_t, dims_t>> {
  public:
    void SetUp() override {
        auto &&[typecode, l_shape, r_shape] = GetParam();

        lhs = hrt::create(typecode, l_shape, host_runtime_tensor::pool_cpu_only)
                  .expect("create tensor failed");
        init_tensor(lhs);

        rhs = hrt::create(typecode, r_shape, host_runtime_tensor::pool_cpu_only)
                  .expect("create tensor failed");
        init_tensor(rhs);
    }

    void TearDown() override {}

  protected:
    runtime_tensor lhs;
    runtime_tensor rhs;
};

INSTANTIATE_TEST_SUITE_P(Normal, NormalTest,
                         testing::Combine(testing::Values(dt_float32),
                                          testing::Values(dims_t{1, 3, 16, 16}),
                                          testing::Values(dims_t{1, 3, 16,
                                                                 16})));

TEST_P(NormalTest, normal) {

    // expected
    int64_t shape_ptr[] = {1, 3, 16, 16};
    auto output_ort = ortki_RandomNormal(1, 0.5f, 1.0f, 1.0f, shape_ptr, 4);
    size_t size = 0;
    void *ptr_ort = tensor_buffer(output_ort, &size);
    dims_t shape(tensor_rank(output_ort));
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(lhs.datatype(), shape,
                                {reinterpret_cast<gsl::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    // actual
    float_t mean_ptr[] = {0.5f};
    float_t scale_ptr[] = {1.0f};
    float_t seed_ptr[] = {1.0f};
    auto mean =
        hrt::create(lhs.datatype(), {1},
                    {reinterpret_cast<gsl::byte *>(mean_ptr), sizeof(mean_ptr)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    auto scale = hrt::create(lhs.datatype(), {1},
                             {reinterpret_cast<gsl::byte *>(scale_ptr),
                              sizeof(scale_ptr)},
                             true, host_runtime_tensor::pool_cpu_only)
                     .expect("create tensor failed");
    auto seed =
        hrt::create(lhs.datatype(), {1},
                    {reinterpret_cast<gsl::byte *>(seed_ptr), sizeof(seed_ptr)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    auto shape0 = hrt::create(dt_int64, {4},
                              {reinterpret_cast<gsl::byte *>(shape_ptr),
                               sizeof(shape_ptr)},
                              true, host_runtime_tensor::pool_cpu_only)
                      .expect("create tensor failed");
    auto output =
        kernels::stackvm::normal(dt_float32, mean.impl(), scale.impl(),
                                 seed.impl(), shape0.impl())
            .expect("normal failed");
    runtime_tensor actual(output.as<tensor>().expect("as tensor failed"));

    bool result = cosine_similarity_tensor(expected, actual) ||
                  is_same_tensor(expected, actual);

    if (!result) {
        print_runtime_tensor(actual);
        print_runtime_tensor(expected);
    }

    // compare
    EXPECT_TRUE(result);
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}