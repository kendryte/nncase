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

#define TEST_CASE_NAME "test_resize_image"

class ResizeImageTest
    : public KernelTest,
      public ::testing::TestWithParam<std::tuple<int>> {
  public:
    void SetUp() override {
        READY_SUBCASE()

        auto typecode = GetDataType("lhs_type");
        auto l_shape = GetShapeArray("i_shape");

        lhs = hrt::create(typecode, l_shape, host_runtime_tensor::pool_cpu_only)
                  .expect("create tensor failed");
        init_tensor(lhs);
    }

    void TearDown() override { CLEAR_SUBCASE() }

  protected:
    runtime_tensor lhs;
};

INSTANTIATE_TEST_SUITE_P(ResizeImage, ResizeImageTest,
                         testing::Combine(testing::Range(0, 1)));

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
        std::cout << "actual ";
        print_runtime_tensor(actual);
        std::cout << "expected ";
        print_runtime_tensor(expected);
    }

    // compare
    EXPECT_TRUE(result);

    // actual

    // actual
    int64_t new_shape_array1[] = {1, 3, 112, 112};
    auto new_shape1 =
        hrt::create(dt_int64, {4},
                    {reinterpret_cast<gsl::byte *>(new_shape_array1),
                     sizeof(new_shape_array1)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    float_t roi_array1[1];
    auto roi1 = hrt::create(dt_float32, {1},
                            {reinterpret_cast<gsl::byte *>(roi_array1),
                             sizeof(roi_array1)},
                            true, host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");
    int32_t exclude_outside_array1[] = {0};

    auto exclude_outside1 =
        hrt::create(dt_int32, {1},
                    {reinterpret_cast<gsl::byte *>(exclude_outside_array1),
                     sizeof(exclude_outside_array1)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    float_t cubic_coeff_a_array1[] = {-0.75f};
    auto cubic_coeff_a1 =
        hrt::create(dt_float32, {1},
                    {reinterpret_cast<gsl::byte *>(cubic_coeff_a_array1),
                     sizeof(cubic_coeff_a_array1)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    float_t extrapolation_value_array1[] = {0.0f};
    auto extrapolation_value1 =
        hrt::create(dt_float32, {1},
                    {reinterpret_cast<gsl::byte *>(extrapolation_value_array1),
                     sizeof(extrapolation_value_array1)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    auto output1 =
        kernels::stackvm::resize_image(
            runtime::stackvm::image_resize_mode_t::bilinear,
            runtime::stackvm::image_resize_transformation_mode_t::
                pytorch_half_pixel,
            runtime::stackvm::image_resize_nearest_mode_t::round_prefer_ceil,
            false, lhs.impl(), roi1.impl(), new_shape1.impl(),
            cubic_coeff_a1.impl(), exclude_outside1.impl(),
            extrapolation_value1.impl())
            .expect("resize_image failed");
    runtime_tensor actual1(output1.as<tensor>().expect("as tensor failed"));

    const char *transformation_mode1 = "pytorch_half_pixel";
    const char *resize_mode_t1 = "linear";
    const char *nearest_mode_t1 = "round_prefer_floor";

    // expected
    auto output_ort1 = ortki_ResizeWithSizes(
        runtime_tensor_2_ort_tensor(lhs), runtime_tensor_2_ort_tensor(roi1),
        runtime_tensor_2_ort_tensor(new_shape1), transformation_mode1, -0.75f,
        0l, 0.0f, resize_mode_t1, nearest_mode_t1);

    size_t size1 = 0;
    void *ptr_ort1 = tensor_buffer(output_ort1, &size1);
    dims_t shape1(tensor_rank(output_ort1));
    tensor_shape(output_ort1, reinterpret_cast<int64_t *>(shape1.data()));
    auto expected1 =
        hrt::create(lhs.datatype(), shape1,
                    {reinterpret_cast<gsl::byte *>(ptr_ort1), size1}, true,
                    host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    bool result1 = is_same_tensor(expected1, actual1) ||
                   cosine_similarity_tensor(expected1, actual1);

    if (!result1) {
        std::cout << "actual ";
        print_runtime_tensor(actual1);
        std::cout << "expected ";
        print_runtime_tensor(expected1);
    }

    // compare
    EXPECT_TRUE(result1);

    // actual
    int64_t new_shape_array2[] = {1, 3, 112, 112};
    auto new_shape2 =
        hrt::create(dt_int64, {4},
                    {reinterpret_cast<gsl::byte *>(new_shape_array2),
                     sizeof(new_shape_array2)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    float_t roi_array2[1];
    auto roi2 = hrt::create(dt_float32, {1},
                            {reinterpret_cast<gsl::byte *>(roi_array2),
                             sizeof(roi_array2)},
                            true, host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");
    int32_t exclude_outside_array2[] = {0};

    auto exclude_outside2 =
        hrt::create(dt_int32, {1},
                    {reinterpret_cast<gsl::byte *>(exclude_outside_array2),
                     sizeof(exclude_outside_array2)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    float_t cubic_coeff_a_array2[] = {-0.75f};
    auto cubic_coeff_a2 =
        hrt::create(dt_float32, {1},
                    {reinterpret_cast<gsl::byte *>(cubic_coeff_a_array2),
                     sizeof(cubic_coeff_a_array2)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    float_t extrapolation_value_array2[] = {0.0f};
    auto extrapolation_value2 =
        hrt::create(dt_float32, {1},
                    {reinterpret_cast<gsl::byte *>(extrapolation_value_array2),
                     sizeof(extrapolation_value_array2)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    auto output2 =
        kernels::stackvm::resize_image(
            runtime::stackvm::image_resize_mode_t::bilinear,
            runtime::stackvm::image_resize_transformation_mode_t::
                pytorch_half_pixel,
            runtime::stackvm::image_resize_nearest_mode_t::round_prefer_ceil,
            false, lhs.impl(), roi2.impl(), new_shape2.impl(),
            cubic_coeff_a2.impl(), exclude_outside2.impl(),
            extrapolation_value2.impl())
            .expect("resize_image failed");
    runtime_tensor actual2(output2.as<tensor>().expect("as tensor failed"));

    const char *transformation_mode2 = "pytorch_half_pixel";
    const char *resize_mode_t2 = "linear";
    const char *nearest_mode_t2 = "round_prefer_ceil";

    // expected
    auto output_ort2 = ortki_ResizeWithSizes(
        runtime_tensor_2_ort_tensor(lhs), runtime_tensor_2_ort_tensor(roi2),
        runtime_tensor_2_ort_tensor(new_shape2), transformation_mode2, -0.75f,
        0l, 0.0f, resize_mode_t2, nearest_mode_t2);

    size_t size2 = 0;
    void *ptr_ort2 = tensor_buffer(output_ort2, &size2);
    dims_t shape2(tensor_rank(output_ort2));
    tensor_shape(output_ort2, reinterpret_cast<int64_t *>(shape2.data()));
    auto expected2 =
        hrt::create(lhs.datatype(), shape2,
                    {reinterpret_cast<gsl::byte *>(ptr_ort2), size2}, true,
                    host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    bool result2 = is_same_tensor(expected2, actual2) ||
                   cosine_similarity_tensor(expected2, actual2);

    if (!result2) {
        std::cout << "actual ";
        print_runtime_tensor(actual2);
        std::cout << "expected ";
        print_runtime_tensor(expected2);
    }

    // compare
    EXPECT_TRUE(result2);
}

int main(int argc, char *argv[]) {
    READY_TEST_CASE_GENERATE()
    FOR_LOOP(lhs_type, i)
    FOR_LOOP(i_shape, j)
    SPLIT_ELEMENT(lhs_type, i)
    SPLIT_ELEMENT(i_shape, j)
    WRITE_SUB_CASE()
    FOR_LOOP_END()
    FOR_LOOP_END()

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}