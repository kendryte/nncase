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

class UniformLikeTest : public KernelTest,
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

INSTANTIATE_TEST_SUITE_P(UniformLike, UniformLikeTest,
                         testing::Combine(testing::Values(dt_float32),
                                          testing::Values(dims_t{1, 3, 16, 16}
                                                          /*dims_t { 3, 16, 16
                                                          }, dims_t { 16, 16 },
                                                          dims_t { 16 },
                                                          dims_t{1}*/),
                                          testing::Values(dims_t{1, 3, 16, 16},
                                                          /*dims_t { 3, 16, 16
                                                          }, dims_t { 16, 16 },
                                                          dims_t { 16 },*/
                                                          dims_t{1})));

TEST_P(UniformLikeTest, UniformLike) {
    auto l_ort = runtime_tensor_2_ort_tensor(lhs);

    // expected
    int64_t shape_u_array[] = {1, 3, 16, 16};
    auto output_ort = ortki_RandomUniformLike(l_ort, 1, 1.0f, 0.0f, 1.0f);
    size_t size = 0;
    void *ptr_ort = tensor_buffer(output_ort, &size);
    dims_t shape(tensor_rank(output_ort));
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(lhs.datatype(), shape,
                                {reinterpret_cast<gsl::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    // actual
    float high_array[] = {1.0f};
    auto high = hrt::create(dt_float32, {1},
                            {reinterpret_cast<gsl::byte *>(high_array), size},
                            true, host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");
    float low_array[] = {0.0f};
    auto low = hrt::create(dt_float32, {1},
                           {reinterpret_cast<gsl::byte *>(low_array), size},
                           true, host_runtime_tensor::pool_cpu_only)
                   .expect("create tensor failed");
    float seed_array[] = {1.0f};
    auto seed = hrt::create(dt_float32, {1},
                            {reinterpret_cast<gsl::byte *>(seed_array), size},
                            true, host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");
    auto shape_u = hrt::create(dt_int32, {4},
                               {reinterpret_cast<gsl::byte *>(shape_u_array), size},
                               true, host_runtime_tensor::pool_cpu_only)
                       .expect("create tensor failed");
    auto output =
        kernels::stackvm::uniform_like(dt_float32, lhs.impl(), high.impl(), low.impl(), seed.impl())
            .expect("uniform_like failed");
    runtime_tensor actual(output.as<tensor>().expect("as tensor failed"));

    // compare
    EXPECT_TRUE(is_same_tensor(expected, actual));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}