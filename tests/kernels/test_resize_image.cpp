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
using namespace nncase::runtime::stackvm;
using namespace ortki;

#define TEST_CASE_NAME "test_resize_image"

class ResizeImageTest : public KernelTest,
                        public ::testing::TestWithParam<std::tuple<int>> {
  public:
    void SetUp() override {
        READY_SUBCASE()

        auto typecode = GetDataType("lhs_type");
        auto l_shape = GetShapeArray("i_shape");
        resize_mode_str = GetString("resize_mode");
        tranform_mode_str = GetString("transform_mode");
        nearest_mode_str = GetString("nearest_mode");
        resize_mode = Str2Mode(resize_mode_str, str_2_resizemode);
        tranform_mode = Str2Mode(tranform_mode_str, str_2_transformmode);
        nearest_mode = Str2Mode(nearest_mode_str, str_2_nearestmode);
        new_shape_array = GetDataArray("o_shape");
        new_shape =
            hrt::create(dt_int64, {4},
                        {reinterpret_cast<gsl::byte *>(new_shape_array.data()),
                         sizeof(new_shape_array[0]) * new_shape_array.size()},
                        true, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
        lhs = hrt::create(typecode, l_shape, host_runtime_tensor::pool_cpu_only)
                  .expect("create tensor failed");
        init_tensor(lhs);
    }

    template <typename T>
    T Str2Mode(std::string type, std::map<std::string, T> &str_2_mode) {
        std::cout << type << std::endl;
        if (str_2_mode.find(type) != str_2_mode.end()) {
            return str_2_mode[type];
        } else {
            // should not be here
            return static_cast<T>(0);
        }
    }
    void TearDown() override { CLEAR_SUBCASE() }

  protected:
    runtime_tensor lhs;
    runtime_tensor new_shape;
    image_resize_mode_t resize_mode;
    std::string resize_mode_str;
    image_resize_transformation_mode_t tranform_mode;
    std::string tranform_mode_str;
    image_resize_nearest_mode_t nearest_mode;
    std::string nearest_mode_str;
    std::vector<int64_t> new_shape_array;
    std::map<std::string, image_resize_mode_t> str_2_resizemode = {
        {"linear", image_resize_mode_t::bilinear},
        {"nearest", image_resize_mode_t::nearest_neighbor}};
    std::map<std::string, image_resize_transformation_mode_t>
        str_2_transformmode = {
            {"half_pixel", image_resize_transformation_mode_t::half_pixel},
            {"pytorch_half_pixel",
             image_resize_transformation_mode_t::pytorch_half_pixel},
            {"align_corners",
             image_resize_transformation_mode_t::align_corners},
            {"asymmetric", image_resize_transformation_mode_t::asymmetric}};
    std::map<std::string, image_resize_nearest_mode_t> str_2_nearestmode = {
        {"round_prefer_floor", image_resize_nearest_mode_t::round_prefer_floor},
        {"round_prefer_ceil", image_resize_nearest_mode_t::round_prefer_ceil},
        {"floor", image_resize_nearest_mode_t::floor},
        {"ceil", image_resize_nearest_mode_t::ceil}};
};

INSTANTIATE_TEST_SUITE_P(ResizeImage, ResizeImageTest,
                         testing::Combine(testing::Range(0, MAX_CASE_NUM)));

TEST_P(ResizeImageTest, ResizeImage) {
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

    float cubic_coeff_a_array[] = {-0.75f};
    auto cubic_coeff_a =
        hrt::create(dt_float32, {1},
                    {reinterpret_cast<gsl::byte *>(cubic_coeff_a_array),
                     sizeof(cubic_coeff_a_array)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    float extrapolation_value_array[] = {0.0f};
    auto extrapolation_value =
        hrt::create(dt_float32, {1},
                    {reinterpret_cast<gsl::byte *>(extrapolation_value_array),
                     sizeof(extrapolation_value_array)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    auto output =
        kernels::stackvm::resize_image(
            resize_mode, tranform_mode, nearest_mode, false, lhs.impl(),
            roi.impl(), new_shape.impl(), cubic_coeff_a.impl(),
            exclude_outside.impl(), extrapolation_value.impl())
            .expect("resize_image failed");
    runtime_tensor actual(output.as<tensor>().expect("as tensor failed"));

    // expected
    auto output_ort = ortki_ResizeWithSizes(
        runtime_tensor_2_ort_tensor(lhs), runtime_tensor_2_ort_tensor(roi),
        runtime_tensor_2_ort_tensor(new_shape), tranform_mode_str.c_str(),
        -0.75f, 0l, 0.0f, resize_mode_str.c_str(), nearest_mode_str.c_str());

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
}

int main(int argc, char *argv[]) {
    READY_TEST_CASE_GENERATE()
    FOR_LOOP(lhs_type, i)
    FOR_LOOP(i_shape, j)
    FOR_LOOP(o_shape, p)
    FOR_LOOP(resize_mode, k)
    FOR_LOOP(transform_mode, m)
    FOR_LOOP(nearest_mode, n)
    SPLIT_ELEMENT(lhs_type, i)
    SPLIT_ELEMENT(i_shape, j)
    SPLIT_ELEMENT(o_shape, p)
    SPLIT_ELEMENT(resize_mode, k)
    SPLIT_ELEMENT(transform_mode, m)
    SPLIT_ELEMENT(nearest_mode, n)
    WRITE_SUB_CASE()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}