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

class ResizeImageTest
    : public KernelTest,
      public ::testing::TestWithParam<std::tuple<nncase::typecode_t, dims_t>> {
  public:
    void SetUp() override {
        auto &&[typecode, l_shape] = GetParam();

        lhs = hrt::create(typecode, l_shape, host_runtime_tensor::pool_cpu_only)
                  .expect("create tensor failed");
        init_tensor(lhs);
    }

    void TearDown() override {}

  protected:
    runtime_tensor lhs;
};

INSTANTIATE_TEST_SUITE_P(ResizeImage, ResizeImageTest,
                         testing::Combine(testing::Values(dt_float32),
                                          testing::Values(dims_t{1, 3, 224,
                                                                 224})));

TEST_P(ResizeImageTest, ResizeImage) {

    // actual
    int64_t new_shape_array[] = {1, 3, 112, 112};
    auto new_shape =
        hrt::create(dt_int64, {4},
                    {reinterpret_cast<gsl::byte *>(new_shape_array),
                     sizeof(new_shape_array)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    float_t roi_array[1];
    auto roi = hrt::create(dt_float32, {1},
                           {reinterpret_cast<gsl::byte *>(roi_array),
                            sizeof(roi_array)},
                           true, host_runtime_tensor::pool_cpu_only)
                   .expect("create tensor failed");
    int32_t exclude_outside_array[] = {0};

    auto exclude_outside =
        hrt::create(dt_int32, {1},
                    {reinterpret_cast<gsl::byte *>(exclude_outside_array),
                     sizeof(exclude_outside_array)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    float_t cubic_coeff_a_array[] = {-0.75f};
    auto cubic_coeff_a =
        hrt::create(dt_float32, {1},
                    {reinterpret_cast<gsl::byte *>(cubic_coeff_a_array),
                     sizeof(cubic_coeff_a_array)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    float_t extrapolation_value_array[] = {0.0f};
    auto extrapolation_value =
        hrt::create(dt_float32, {1},
                    {reinterpret_cast<gsl::byte *>(extrapolation_value_array),
                     sizeof(extrapolation_value_array)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    auto output =
        kernels::stackvm::resize_image(
            runtime::stackvm::image_resize_mode_t::bilinear,
            runtime::stackvm::image_resize_transformation_mode_t::half_pixel,
            runtime::stackvm::image_resize_nearest_mode_t::round_prefer_floor,
            false, lhs.impl(), roi.impl(), new_shape.impl(),
            cubic_coeff_a.impl(), exclude_outside.impl(),
            extrapolation_value.impl())
            .expect("resize_image failed");
    runtime_tensor actual(output.as<tensor>().expect("as tensor failed"));

    const char *transformation_mode = "half_pixel";
    const char *resize_mode_t = "linear";
    const char *nearest_mode_t = "round_prefer_floor";

    // expected
    auto output_ort = ortki_ResizeWithSizes(
        runtime_tensor_2_ort_tensor(lhs), runtime_tensor_2_ort_tensor(roi),
        runtime_tensor_2_ort_tensor(new_shape), transformation_mode, -0.75f, 0l,
        0.0f, resize_mode_t, nearest_mode_t);

    size_t size = 0;
    void *ptr_ort = tensor_buffer(output_ort, &size);
    dims_t shape(tensor_rank(output_ort));
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(lhs.datatype(), shape,
                                {reinterpret_cast<gsl::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");
    bool result = is_same_tensor(expected, actual) ||
                  cosine_similarity_tensor(expected, actual);

    if (!result) {
        std::cout<<"actual ";
        print_runtime_tensor(actual);
        std::cout<<"expected ";
        print_runtime_tensor(expected);
    }

    // compare
    EXPECT_TRUE(result);
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}