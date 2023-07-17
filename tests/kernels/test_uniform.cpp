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

class UniformTest
    : public KernelTest,
      public ::testing::TestWithParam<std::tuple<nncase::typecode_t, dims_t>> {
  public:
    void SetUp() override {
        auto &&[typecode, l_shape] = GetParam();

        input =
            hrt::create(typecode, l_shape, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
        init_tensor(input);
    }

    void TearDown() override {}

  protected:
    runtime_tensor input;
};

INSTANTIATE_TEST_SUITE_P(Uniform, UniformTest,
                         testing::Combine(testing::Values(dt_float32),
                                          testing::Values(dims_t{1, 3, 16, 16},
                                                          /*dims_t { 3, 16, 16
                                                          }, dims_t { 16, 16 },
                                                          dims_t { 16 },*/
                                                          dims_t{1})));

TEST_P(UniformTest, Uniform) {
    [[maybe_unused]] auto l_ort = runtime_tensor_2_ort_tensor(input);

    // expected
    int64_t shape_u_array[] = {1, 3, 16, 16};
    auto output_ort =
        ortki_RandomUniform(1, 1.0f, 0.0f, 1.0f, shape_u_array, 4);
    size_t size = 0;
    void *ptr_ort = tensor_buffer(output_ort, &size);
    dims_t shape(tensor_rank(output_ort));
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(input.datatype(), shape,
                                {reinterpret_cast<gsl::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    // actual
    float_t high_array[] = {1.0f};
    auto high = hrt::create(dt_float32, {1},
                            {reinterpret_cast<gsl::byte *>(high_array),
                             sizeof(high_array)},
                            true, host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");
    float_t low_array[] = {0.0f};
    auto low = hrt::create(dt_float32, {1},
                           {reinterpret_cast<gsl::byte *>(low_array),
                            sizeof(low_array)},
                           true, host_runtime_tensor::pool_cpu_only)
                   .expect("create tensor failed");
    float_t seed_array[] = {1.0f};
    auto seed = hrt::create(dt_float32, {1},
                            {reinterpret_cast<gsl::byte *>(seed_array),
                             sizeof(seed_array)},
                            true, host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");
    auto shape_u = hrt::create(dt_int64, {4},
                               {reinterpret_cast<gsl::byte *>(shape_u_array),
                                sizeof(shape_u_array)},
                               true, host_runtime_tensor::pool_cpu_only)
                       .expect("create tensor failed");
    auto output = kernels::stackvm::uniform(dt_float32, high.impl(), low.impl(),
                                            seed.impl(), shape_u.impl())
                      .expect("uniform failed");
    runtime_tensor actual(output.as<tensor>().expect("as tensor failed"));

    bool result = is_same_tensor(expected, actual) ||
                  cosine_similarity_tensor(expected, actual);

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