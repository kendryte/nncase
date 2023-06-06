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

class ResizeImageTest : public KernelTest,
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

INSTANTIATE_TEST_SUITE_P(ResizeImage, ResizeImageTest,
                         testing::Combine(testing::Values(dt_float32, dt_int32,
                                                          dt_int64),
                                          testing::Values(dims_t{1, 3, 16, 16},
                                                          /*dims_t { 3, 16, 16
                                                          }, dims_t { 16, 16 },
                                                          dims_t { 16 },*/
                                                          dims_t{1}),
                                          testing::Values(dims_t{1, 3, 16, 16},
                                                          /*dims_t { 3, 16, 16
                                                          }, dims_t { 16, 16 },
                                                          dims_t { 16 },*/
                                                          dims_t{1})));

TEST_P(ResizeImageTest, ResizeImage) {
    //    auto l_ort = runtime_tensor_2_ort_tensor(lhs);
    //    auto r_ort = runtime_tensor_2_ort_tensor(rhs);

    // expected
    //    size_t size = 0;
    int32_t expected_array[] = {1, 3, 112, 112};
    auto expected =
        hrt::create(dt_float32, {4},
                    {reinterpret_cast<gsl::byte *>(expected_array), 16}, true,
                    host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    // actual
    float_t roi_array[1];
    auto roi = hrt::create(dt_float32, {1},
                           {reinterpret_cast<gsl::byte *>(roi_array), 4}, true,
                           host_runtime_tensor::pool_cpu_only)
                   .expect("create tensor failed");
    bool exclude_outside_array[] = {false};
    auto exclude_outside =
        hrt::create(dt_boolean, {1},
                    {reinterpret_cast<gsl::byte *>(exclude_outside_array), 1},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    float_t cubic_coeff_a_array[] = {-0.75f};
    auto cubic_coeff_a =
        hrt::create(dt_float32, {1},
                    {reinterpret_cast<gsl::byte *>(cubic_coeff_a_array), 4},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    float_t extrapolation_value_array[] = {-0.0f};
    auto extrapolation_value =
        hrt::create(
            dt_float32, {1},
            {reinterpret_cast<gsl::byte *>(extrapolation_value_array), 4}, true,
            host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    //    auto output =
    //        kernels::stackvm::resize_image(
    //            runtime::stackvm::image_resize_mode_t::bilinear,
    //            runtime::stackvm::image_resize_transformation_mode_t::asymmetric,
    //            runtime::stackvm::image_resize_nearest_mode_t::floor, false,
    //            lhs.impl(), roi.impl(), expected.impl(), cubic_coeff_a.impl(),
    //            exclude_outside.impl(), extrapolation_value.impl())
    //            .expect("resize_image failed");
    //    runtime_tensor actual(output.as<tensor>().expect("as tensor failed"));

    // compare
    //    EXPECT_TRUE(is_same_tensor(expected, actual));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}