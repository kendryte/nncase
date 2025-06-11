/* Copyright 2019-2024 Canaan Inc.
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
#include "nncase/ntt/shape.h"
#include "nncase/ntt/tensor.h"
#include "nncase/ntt/tensor_traits.h"
#include "nncase/ntt/vector.h"
#include "ntt_test.h"
#include "ortki_helper.h"
#include <gtest/gtest.h>
#include <iostream>
#include <nncase/ntt/ntt.h>
#include <ortki/operators.h>

using namespace nncase;
using namespace ortki;


TEST(PackTestFloat, fixed_1D_vector_contiguous_pack_axis_2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t C = 1;
    constexpr size_t H = 77;
    constexpr size_t W_coefficient = 3;
    constexpr size_t W = W_coefficient * P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    alignas(32) auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<C, H, W>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<C, H, W / P>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // ORT reference implementation
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t reshape_data[] = {(int64_t)C, (int64_t)H, (int64_t)(W / P), (int64_t)P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 3};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<C, H, W / P>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_1D_vector_contiguous_pack_axis_1_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t C = 1;
    constexpr size_t H_coefficient = 77;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W = 3;
    float min_input = -10.0f;
    float max_input = 10.0f;

    alignas(32) auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<C, H, W>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<C, H / P, W>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<1>);

    // ORT reference implementation
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t reshape_data[] = {(int64_t)C, (int64_t)(H / P), (int64_t)P, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 3, 2};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<C, H / P, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_1D_vector_contiguous_pack_axis_0_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t C_coefficient = 1;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H = 77;
    constexpr size_t W = 3;
    float min_input = -10.0f;
    float max_input = 10.0f;

    alignas(32) auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<C, H, W>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<C / P, H, W>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<0>);

    // ORT reference implementation
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t reshape_data[] = {(int64_t)(C / P), (int64_t)P, (int64_t)H, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 2, 3, 1};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<C / P, H, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_1D_vector_non_contiguous_dim1_mul2_pack_axis_2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t C = 1;
    constexpr size_t H = 77;
    constexpr size_t W_coefficient = 3;
    constexpr size_t W = W_coefficient * P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::fixed_shape_v<C, (H) *2, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<C, H, W>,
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<C, H, W / P>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::fixed_shape_v<C, H, W>);
    
    for (size_t c = 0; c < C; c++) {
        for (size_t h = 0; h < H; h++) {
            for (size_t w = 0; w < W; w++) {
                continuous_input(c, h, w) = ntt_input(c, h, w);
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)C, (int64_t)H, (int64_t)(W / P), (int64_t)P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 3};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<C, H, W / P>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_1D_vector_non_contiguous_dim1_mul2_pack_axis_1_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t C = 1;
    constexpr size_t H_coefficient = 77;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W = 3;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::fixed_shape_v<C, (H) *2, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<C, H, W>,
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<C, H / P, W>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<1>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::fixed_shape_v<C, H, W>);
    
    for (size_t c = 0; c < C; c++) {
        for (size_t h = 0; h < H; h++) {
            for (size_t w = 0; w < W; w++) {
                continuous_input(c, h, w) = ntt_input(c, h, w);
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)C, (int64_t)(H / P), (int64_t)P, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 3, 2};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<C, H / P, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_1D_vector_non_contiguous_dim1_mul2_pack_axis_0_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t C_coefficient = 1;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H = 77;
    constexpr size_t W = 3;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::fixed_shape_v<C, (H) *2, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<C, H, W>,
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<C / P, H, W>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<0>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::fixed_shape_v<C, H, W>);
    
    for (size_t c = 0; c < C; c++) {
        for (size_t h = 0; h < H; h++) {
            for (size_t w = 0; w < W; w++) {
                continuous_input(c, h, w) = ntt_input(c, h, w);
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)(C / P), (int64_t)P, (int64_t)H, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 2, 3, 1};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<C / P, H, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_2D_vector_contiguous_pack_axis_0_1_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t C_coefficient = 1;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H_coefficient = 77;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W = 3;
    float min_input = -10.0f;
    float max_input = 10.0f;

    alignas(32) auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<C, H, W>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<C / P, H / P, W>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<0, 1>);

    // ORT reference implementation
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t reshape_data[] = {(int64_t)(C / P), (int64_t)P, (int64_t)(H / P), (int64_t)P, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 2, 4, 1, 3};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<C / P, H / P, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_2D_vector_contiguous_pack_axis_1_2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t C = 1;
    constexpr size_t H_coefficient = 77;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W_coefficient = 3;
    constexpr size_t W = W_coefficient * P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    alignas(32) auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<C, H, W>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<C, H / P, W / P>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<1, 2>);

    // ORT reference implementation
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t reshape_data[] = {(int64_t)C, (int64_t)(H / P), (int64_t)P, (int64_t)(W / P), (int64_t)P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 3, 2, 4};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<C, H / P, W / P>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_2D_vector_non_contiguous_dim1_mul2_pack_axis_0_1_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t C_coefficient = 1;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H_coefficient = 77;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W = 3;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::fixed_shape_v<C, (H) *2, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<C, H, W>,
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<C / P, H / P, W>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<0, 1>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::fixed_shape_v<C, H, W>);
    
    for (size_t c = 0; c < C; c++) {
        for (size_t h = 0; h < H; h++) {
            for (size_t w = 0; w < W; w++) {
                continuous_input(c, h, w) = ntt_input(c, h, w);
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)(C / P), (int64_t)P, (int64_t)(H / P), (int64_t)P, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 2, 4, 1, 3};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<C / P, H / P, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_2D_vector_non_contiguous_dim1_mul2_pack_axis_1_2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t C = 1;
    constexpr size_t H_coefficient = 77;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W_coefficient = 3;
    constexpr size_t W = W_coefficient * P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::fixed_shape_v<C, (H) *2, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<C, H, W>,
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<C, H / P, W / P>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<1, 2>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::fixed_shape_v<C, H, W>);
    
    for (size_t c = 0; c < C; c++) {
        for (size_t h = 0; h < H; h++) {
            for (size_t w = 0; w < W; w++) {
                continuous_input(c, h, w) = ntt_input(c, h, w);
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)C, (int64_t)(H / P), (int64_t)P, (int64_t)(W / P), (int64_t)P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 3, 2, 4};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<C, H / P, W / P>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_1D_vector_contiguous_pack_axis_2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t C = 1;
    constexpr size_t H = 77;
    constexpr size_t W_coefficient = 3;
    constexpr size_t W = W_coefficient * P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    alignas(32) auto ntt_input = ntt::make_tensor<float>(ntt::make_shape(C, H, W));
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(C, H, W / P));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // ORT reference implementation
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t reshape_data[] = {(int64_t)C, (int64_t)H, (int64_t)(W / P), (int64_t)P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 3};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(C, H, W / P));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_1D_vector_contiguous_pack_axis_1_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t C = 1;
    constexpr size_t H_coefficient = 77;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W = 3;
    float min_input = -10.0f;
    float max_input = 10.0f;

    alignas(32) auto ntt_input = ntt::make_tensor<float>(ntt::make_shape(C, H, W));
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(C, H / P, W));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<1>);

    // ORT reference implementation
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t reshape_data[] = {(int64_t)C, (int64_t)(H / P), (int64_t)P, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 3, 2};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(C, H / P, W));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_1D_vector_contiguous_pack_axis_0_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t C_coefficient = 1;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H = 77;
    constexpr size_t W = 3;
    float min_input = -10.0f;
    float max_input = 10.0f;

    alignas(32) auto ntt_input = ntt::make_tensor<float>(ntt::make_shape(C, H, W));
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(C / P, H, W));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<0>);

    // ORT reference implementation
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t reshape_data[] = {(int64_t)(C / P), (int64_t)P, (int64_t)H, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 2, 3, 1};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(C / P, H, W));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_1D_vector_non_contiguous_dim1_mul2_pack_axis_2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t C = 1;
    constexpr size_t H = 77;
    constexpr size_t W_coefficient = 3;
    constexpr size_t W = W_coefficient * P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::make_shape(C, (H) *2, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::make_shape(C, H, W),
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(C, H, W / P));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::make_shape(C, H, W));
    
    for (size_t c = 0; c < C; c++) {
        for (size_t h = 0; h < H; h++) {
            for (size_t w = 0; w < W; w++) {
                continuous_input(c, h, w) = ntt_input(c, h, w);
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)C, (int64_t)H, (int64_t)(W / P), (int64_t)P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 3};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(C, H, W / P));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_1D_vector_non_contiguous_dim1_mul2_pack_axis_1_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t C = 1;
    constexpr size_t H_coefficient = 77;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W = 3;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::make_shape(C, (H) *2, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::make_shape(C, H, W),
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(C, H / P, W));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<1>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::make_shape(C, H, W));
    
    for (size_t c = 0; c < C; c++) {
        for (size_t h = 0; h < H; h++) {
            for (size_t w = 0; w < W; w++) {
                continuous_input(c, h, w) = ntt_input(c, h, w);
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)C, (int64_t)(H / P), (int64_t)P, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 3, 2};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(C, H / P, W));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_1D_vector_non_contiguous_dim1_mul2_pack_axis_0_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t C_coefficient = 1;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H = 77;
    constexpr size_t W = 3;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::make_shape(C, (H) *2, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::make_shape(C, H, W),
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(C / P, H, W));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<0>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::make_shape(C, H, W));
    
    for (size_t c = 0; c < C; c++) {
        for (size_t h = 0; h < H; h++) {
            for (size_t w = 0; w < W; w++) {
                continuous_input(c, h, w) = ntt_input(c, h, w);
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)(C / P), (int64_t)P, (int64_t)H, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 2, 3, 1};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(C / P, H, W));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_2D_vector_contiguous_pack_axis_0_1_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t C_coefficient = 1;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H_coefficient = 77;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W = 3;
    float min_input = -10.0f;
    float max_input = 10.0f;

    alignas(32) auto ntt_input = ntt::make_tensor<float>(ntt::make_shape(C, H, W));
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(C / P, H / P, W));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<0, 1>);

    // ORT reference implementation
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t reshape_data[] = {(int64_t)(C / P), (int64_t)P, (int64_t)(H / P), (int64_t)P, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 2, 4, 1, 3};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(C / P, H / P, W));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_2D_vector_contiguous_pack_axis_1_2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t C = 1;
    constexpr size_t H_coefficient = 77;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W_coefficient = 3;
    constexpr size_t W = W_coefficient * P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    alignas(32) auto ntt_input = ntt::make_tensor<float>(ntt::make_shape(C, H, W));
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(C, H / P, W / P));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<1, 2>);

    // ORT reference implementation
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t reshape_data[] = {(int64_t)C, (int64_t)(H / P), (int64_t)P, (int64_t)(W / P), (int64_t)P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 3, 2, 4};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(C, H / P, W / P));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_2D_vector_non_contiguous_dim1_mul2_pack_axis_0_1_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t C_coefficient = 1;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H_coefficient = 77;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W = 3;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::make_shape(C, (H) *2, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::make_shape(C, H, W),
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(C / P, H / P, W));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<0, 1>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::make_shape(C, H, W));
    
    for (size_t c = 0; c < C; c++) {
        for (size_t h = 0; h < H; h++) {
            for (size_t w = 0; w < W; w++) {
                continuous_input(c, h, w) = ntt_input(c, h, w);
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)(C / P), (int64_t)P, (int64_t)(H / P), (int64_t)P, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 2, 4, 1, 3};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(C / P, H / P, W));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_2D_vector_non_contiguous_dim1_mul2_pack_axis_1_2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t C = 1;
    constexpr size_t H_coefficient = 77;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W_coefficient = 3;
    constexpr size_t W = W_coefficient * P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::make_shape(C, (H) *2, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::make_shape(C, H, W),
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(C, H / P, W / P));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<1, 2>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::make_shape(C, H, W));
    
    for (size_t c = 0; c < C; c++) {
        for (size_t h = 0; h < H; h++) {
            for (size_t w = 0; w < W; w++) {
                continuous_input(c, h, w) = ntt_input(c, h, w);
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)C, (int64_t)(H / P), (int64_t)P, (int64_t)(W / P), (int64_t)P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 3, 2, 4};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(C, H / P, W / P));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_1D_vector_contiguous_pack_axis_3_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W_coefficient = 4;
    constexpr size_t W = W_coefficient * P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    alignas(32) auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C, H, W / P>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    // ORT reference implementation
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)H, (int64_t)(W / P), (int64_t)P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 3, 4};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C, H, W / P>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_1D_vector_contiguous_pack_axis_2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H_coefficient = 4;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    alignas(32) auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C, H / P, W>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // ORT reference implementation
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)(H / P), (int64_t)P, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 4, 3};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C, H / P, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_1D_vector_contiguous_pack_axis_1_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C_coefficient = 8;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    alignas(32) auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C / P, H, W>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<1>);

    // ORT reference implementation
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)(C / P), (int64_t)P, (int64_t)H, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 3, 4, 2};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C / P, H, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_1D_vector_contiguous_pack_axis_0_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N_coefficient = 2;
    constexpr size_t N = N_coefficient * P;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    alignas(32) auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N / P, C, H, W>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<0>);

    // ORT reference implementation
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t reshape_data[] = {(int64_t)(N / P), (int64_t)P, (int64_t)C, (int64_t)H, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 2, 3, 4, 1};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N / P, C, H, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_1D_vector_non_contiguous_dim2_add5_pack_axis_3_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W_coefficient = 4;
    constexpr size_t W = W_coefficient * P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 2)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, (H) +7, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C, H, W / P>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)H, (int64_t)(W / P), (int64_t)P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 3, 4};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C, H, W / P>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_1D_vector_non_contiguous_dim2_add5_pack_axis_2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H_coefficient = 4;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 2)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, (H) +7, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C, H / P, W>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)(H / P), (int64_t)P, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 4, 3};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C, H / P, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_1D_vector_non_contiguous_dim2_add5_pack_axis_1_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C_coefficient = 8;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 2)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, (H) +7, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C / P, H, W>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<1>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)(C / P), (int64_t)P, (int64_t)H, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 3, 4, 2};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C / P, H, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_1D_vector_non_contiguous_dim2_add5_pack_axis_0_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N_coefficient = 2;
    constexpr size_t N = N_coefficient * P;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 2)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, (H) +7, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N / P, C, H, W>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<0>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)(N / P), (int64_t)P, (int64_t)C, (int64_t)H, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 2, 3, 4, 1};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N / P, C, H, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_1D_vector_non_contiguous_dim2_mul2_pack_axis_3_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W_coefficient = 4;
    constexpr size_t W = W_coefficient * P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 2)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, (H) *2, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C, H, W / P>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)H, (int64_t)(W / P), (int64_t)P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 3, 4};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C, H, W / P>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_1D_vector_non_contiguous_dim2_mul2_pack_axis_2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H_coefficient = 4;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 2)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, (H) *2, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C, H / P, W>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)(H / P), (int64_t)P, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 4, 3};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C, H / P, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_1D_vector_non_contiguous_dim2_mul2_pack_axis_1_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C_coefficient = 8;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 2)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, (H) *2, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C / P, H, W>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<1>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)(C / P), (int64_t)P, (int64_t)H, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 3, 4, 2};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C / P, H, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_1D_vector_non_contiguous_dim2_mul2_pack_axis_0_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N_coefficient = 2;
    constexpr size_t N = N_coefficient * P;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 2)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, (H) *2, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N / P, C, H, W>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<0>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)(N / P), (int64_t)P, (int64_t)C, (int64_t)H, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 2, 3, 4, 1};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N / P, C, H, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_1D_vector_non_contiguous_dim1_mul2_pack_axis_3_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W_coefficient = 4;
    constexpr size_t W = W_coefficient * P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::fixed_shape_v<N, (C) *2, H, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C, H, W / P>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)H, (int64_t)(W / P), (int64_t)P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 3, 4};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C, H, W / P>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_1D_vector_non_contiguous_dim1_mul2_pack_axis_2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H_coefficient = 4;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::fixed_shape_v<N, (C) *2, H, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C, H / P, W>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)(H / P), (int64_t)P, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 4, 3};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C, H / P, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_1D_vector_non_contiguous_dim1_mul2_pack_axis_1_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C_coefficient = 8;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::fixed_shape_v<N, (C) *2, H, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C / P, H, W>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<1>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)(C / P), (int64_t)P, (int64_t)H, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 3, 4, 2};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C / P, H, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_1D_vector_non_contiguous_dim1_mul2_pack_axis_0_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N_coefficient = 2;
    constexpr size_t N = N_coefficient * P;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::fixed_shape_v<N, (C) *2, H, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N / P, C, H, W>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<0>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)(N / P), (int64_t)P, (int64_t)C, (int64_t)H, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 2, 3, 4, 1};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N / P, C, H, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_1D_vector_non_contiguous_dim1_add5_pack_axis_3_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W_coefficient = 4;
    constexpr size_t W = W_coefficient * P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::fixed_shape_v<N, (C) +7, H, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C, H, W / P>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)H, (int64_t)(W / P), (int64_t)P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 3, 4};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C, H, W / P>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_1D_vector_non_contiguous_dim1_add5_pack_axis_2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H_coefficient = 4;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::fixed_shape_v<N, (C) +7, H, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C, H / P, W>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)(H / P), (int64_t)P, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 4, 3};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C, H / P, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_1D_vector_non_contiguous_dim1_add5_pack_axis_1_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C_coefficient = 8;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::fixed_shape_v<N, (C) +7, H, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C / P, H, W>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<1>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)(C / P), (int64_t)P, (int64_t)H, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 3, 4, 2};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C / P, H, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_1D_vector_non_contiguous_dim1_add5_pack_axis_0_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N_coefficient = 2;
    constexpr size_t N = N_coefficient * P;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::fixed_shape_v<N, (C) +7, H, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N / P, C, H, W>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<0>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)(N / P), (int64_t)P, (int64_t)C, (int64_t)H, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 2, 3, 4, 1};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N / P, C, H, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_2D_vector_contiguous_pack_axis_0_1_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N_coefficient = 2;
    constexpr size_t N = N_coefficient * P;
    constexpr size_t C_coefficient = 8;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    alignas(32) auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N / P, C / P, H, W>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<0, 1>);

    // ORT reference implementation
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t reshape_data[] = {(int64_t)(N / P), (int64_t)P, (int64_t)(C / P), (int64_t)P, (int64_t)H, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 2, 4, 5, 1, 3};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N / P, C / P, H, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_2D_vector_contiguous_pack_axis_1_2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C_coefficient = 8;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H_coefficient = 4;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    alignas(32) auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N, C / P, H / P, W>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<1, 2>);

    // ORT reference implementation
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)(C / P), (int64_t)P, (int64_t)(H / P), (int64_t)P, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 3, 5, 2, 4};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N, C / P, H / P, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_2D_vector_contiguous_pack_axis_2_3_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H_coefficient = 4;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W_coefficient = 4;
    constexpr size_t W = W_coefficient * P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    alignas(32) auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N, C, H / P, W / P>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<2, 3>);

    // ORT reference implementation
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)(H / P), (int64_t)P, (int64_t)(W / P), (int64_t)P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 4, 3, 5};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N, C, H / P, W / P>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_2D_vector_non_contiguous_dim2_add5_pack_axis_0_1_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N_coefficient = 2;
    constexpr size_t N = N_coefficient * P;
    constexpr size_t C_coefficient = 8;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 2)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, (H) +7, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N / P, C / P, H, W>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<0, 1>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)(N / P), (int64_t)P, (int64_t)(C / P), (int64_t)P, (int64_t)H, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 2, 4, 5, 1, 3};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N / P, C / P, H, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_2D_vector_non_contiguous_dim2_add5_pack_axis_1_2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C_coefficient = 8;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H_coefficient = 4;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 2)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, (H) +7, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N, C / P, H / P, W>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<1, 2>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)(C / P), (int64_t)P, (int64_t)(H / P), (int64_t)P, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 3, 5, 2, 4};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N, C / P, H / P, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_2D_vector_non_contiguous_dim2_add5_pack_axis_2_3_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H_coefficient = 4;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W_coefficient = 4;
    constexpr size_t W = W_coefficient * P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 2)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, (H) +7, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N, C, H / P, W / P>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<2, 3>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)(H / P), (int64_t)P, (int64_t)(W / P), (int64_t)P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 4, 3, 5};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N, C, H / P, W / P>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_2D_vector_non_contiguous_dim2_mul2_pack_axis_0_1_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N_coefficient = 2;
    constexpr size_t N = N_coefficient * P;
    constexpr size_t C_coefficient = 8;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 2)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, (H) *2, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N / P, C / P, H, W>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<0, 1>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)(N / P), (int64_t)P, (int64_t)(C / P), (int64_t)P, (int64_t)H, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 2, 4, 5, 1, 3};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N / P, C / P, H, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_2D_vector_non_contiguous_dim2_mul2_pack_axis_1_2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C_coefficient = 8;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H_coefficient = 4;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 2)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, (H) *2, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N, C / P, H / P, W>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<1, 2>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)(C / P), (int64_t)P, (int64_t)(H / P), (int64_t)P, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 3, 5, 2, 4};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N, C / P, H / P, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_2D_vector_non_contiguous_dim2_mul2_pack_axis_2_3_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H_coefficient = 4;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W_coefficient = 4;
    constexpr size_t W = W_coefficient * P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 2)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, (H) *2, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N, C, H / P, W / P>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<2, 3>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)(H / P), (int64_t)P, (int64_t)(W / P), (int64_t)P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 4, 3, 5};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N, C, H / P, W / P>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_2D_vector_non_contiguous_dim1_mul2_pack_axis_0_1_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N_coefficient = 2;
    constexpr size_t N = N_coefficient * P;
    constexpr size_t C_coefficient = 8;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::fixed_shape_v<N, (C) *2, H, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N / P, C / P, H, W>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<0, 1>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)(N / P), (int64_t)P, (int64_t)(C / P), (int64_t)P, (int64_t)H, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 2, 4, 5, 1, 3};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N / P, C / P, H, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_2D_vector_non_contiguous_dim1_mul2_pack_axis_1_2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C_coefficient = 8;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H_coefficient = 4;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::fixed_shape_v<N, (C) *2, H, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N, C / P, H / P, W>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<1, 2>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)(C / P), (int64_t)P, (int64_t)(H / P), (int64_t)P, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 3, 5, 2, 4};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N, C / P, H / P, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_2D_vector_non_contiguous_dim1_mul2_pack_axis_2_3_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H_coefficient = 4;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W_coefficient = 4;
    constexpr size_t W = W_coefficient * P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::fixed_shape_v<N, (C) *2, H, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N, C, H / P, W / P>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<2, 3>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)(H / P), (int64_t)P, (int64_t)(W / P), (int64_t)P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 4, 3, 5};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N, C, H / P, W / P>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_2D_vector_non_contiguous_dim1_add5_pack_axis_0_1_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N_coefficient = 2;
    constexpr size_t N = N_coefficient * P;
    constexpr size_t C_coefficient = 8;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::fixed_shape_v<N, (C) +7, H, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N / P, C / P, H, W>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<0, 1>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)(N / P), (int64_t)P, (int64_t)(C / P), (int64_t)P, (int64_t)H, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 2, 4, 5, 1, 3};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N / P, C / P, H, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_2D_vector_non_contiguous_dim1_add5_pack_axis_1_2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C_coefficient = 8;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H_coefficient = 4;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::fixed_shape_v<N, (C) +7, H, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N, C / P, H / P, W>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<1, 2>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)(C / P), (int64_t)P, (int64_t)(H / P), (int64_t)P, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 3, 5, 2, 4};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N, C / P, H / P, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_2D_vector_non_contiguous_dim1_add5_pack_axis_2_3_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H_coefficient = 4;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W_coefficient = 4;
    constexpr size_t W = W_coefficient * P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::fixed_shape_v<N, (C) +7, H, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N, C, H / P, W / P>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<2, 3>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)(H / P), (int64_t)P, (int64_t)(W / P), (int64_t)P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 4, 3, 5};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N, C, H / P, W / P>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_1D_vector_contiguous_pack_axis_3_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W_coefficient = 4;
    constexpr size_t W = W_coefficient * P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    alignas(32) auto ntt_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C, H, W / P));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    // ORT reference implementation
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)H, (int64_t)(W / P), (int64_t)P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 3, 4};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C, H, W / P));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_1D_vector_contiguous_pack_axis_2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H_coefficient = 4;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    alignas(32) auto ntt_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C, H / P, W));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // ORT reference implementation
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)(H / P), (int64_t)P, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 4, 3};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C, H / P, W));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_1D_vector_contiguous_pack_axis_1_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C_coefficient = 8;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    alignas(32) auto ntt_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C / P, H, W));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<1>);

    // ORT reference implementation
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)(C / P), (int64_t)P, (int64_t)H, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 3, 4, 2};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C / P, H, W));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_1D_vector_contiguous_pack_axis_0_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N_coefficient = 2;
    constexpr size_t N = N_coefficient * P;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    alignas(32) auto ntt_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N / P, C, H, W));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<0>);

    // ORT reference implementation
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t reshape_data[] = {(int64_t)(N / P), (int64_t)P, (int64_t)C, (int64_t)H, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 2, 3, 4, 1};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N / P, C, H, W));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_1D_vector_non_contiguous_dim2_add5_pack_axis_3_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W_coefficient = 4;
    constexpr size_t W = W_coefficient * P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 2)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::make_shape(N, C, (H) +7, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C, H, W / P));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)H, (int64_t)(W / P), (int64_t)P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 3, 4};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C, H, W / P));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_1D_vector_non_contiguous_dim2_add5_pack_axis_2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H_coefficient = 4;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 2)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::make_shape(N, C, (H) +7, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C, H / P, W));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)(H / P), (int64_t)P, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 4, 3};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C, H / P, W));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_1D_vector_non_contiguous_dim2_add5_pack_axis_1_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C_coefficient = 8;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 2)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::make_shape(N, C, (H) +7, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C / P, H, W));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<1>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)(C / P), (int64_t)P, (int64_t)H, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 3, 4, 2};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C / P, H, W));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_1D_vector_non_contiguous_dim2_add5_pack_axis_0_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N_coefficient = 2;
    constexpr size_t N = N_coefficient * P;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 2)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::make_shape(N, C, (H) +7, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N / P, C, H, W));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<0>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)(N / P), (int64_t)P, (int64_t)C, (int64_t)H, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 2, 3, 4, 1};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N / P, C, H, W));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_1D_vector_non_contiguous_dim2_mul2_pack_axis_3_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W_coefficient = 4;
    constexpr size_t W = W_coefficient * P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 2)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::make_shape(N, C, (H) *2, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C, H, W / P));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)H, (int64_t)(W / P), (int64_t)P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 3, 4};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C, H, W / P));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_1D_vector_non_contiguous_dim2_mul2_pack_axis_2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H_coefficient = 4;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 2)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::make_shape(N, C, (H) *2, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C, H / P, W));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)(H / P), (int64_t)P, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 4, 3};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C, H / P, W));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_1D_vector_non_contiguous_dim2_mul2_pack_axis_1_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C_coefficient = 8;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 2)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::make_shape(N, C, (H) *2, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C / P, H, W));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<1>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)(C / P), (int64_t)P, (int64_t)H, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 3, 4, 2};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C / P, H, W));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_1D_vector_non_contiguous_dim2_mul2_pack_axis_0_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N_coefficient = 2;
    constexpr size_t N = N_coefficient * P;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 2)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::make_shape(N, C, (H) *2, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N / P, C, H, W));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<0>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)(N / P), (int64_t)P, (int64_t)C, (int64_t)H, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 2, 3, 4, 1};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N / P, C, H, W));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_1D_vector_non_contiguous_dim1_mul2_pack_axis_3_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W_coefficient = 4;
    constexpr size_t W = W_coefficient * P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::make_shape(N, (C) *2, H, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C, H, W / P));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)H, (int64_t)(W / P), (int64_t)P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 3, 4};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C, H, W / P));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_1D_vector_non_contiguous_dim1_mul2_pack_axis_2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H_coefficient = 4;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::make_shape(N, (C) *2, H, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C, H / P, W));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)(H / P), (int64_t)P, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 4, 3};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C, H / P, W));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_1D_vector_non_contiguous_dim1_mul2_pack_axis_1_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C_coefficient = 8;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::make_shape(N, (C) *2, H, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C / P, H, W));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<1>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)(C / P), (int64_t)P, (int64_t)H, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 3, 4, 2};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C / P, H, W));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_1D_vector_non_contiguous_dim1_mul2_pack_axis_0_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N_coefficient = 2;
    constexpr size_t N = N_coefficient * P;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::make_shape(N, (C) *2, H, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N / P, C, H, W));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<0>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)(N / P), (int64_t)P, (int64_t)C, (int64_t)H, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 2, 3, 4, 1};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N / P, C, H, W));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_1D_vector_non_contiguous_dim1_add5_pack_axis_3_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W_coefficient = 4;
    constexpr size_t W = W_coefficient * P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::make_shape(N, (C) +7, H, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C, H, W / P));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)H, (int64_t)(W / P), (int64_t)P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 3, 4};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C, H, W / P));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_1D_vector_non_contiguous_dim1_add5_pack_axis_2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H_coefficient = 4;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::make_shape(N, (C) +7, H, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C, H / P, W));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)(H / P), (int64_t)P, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 4, 3};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C, H / P, W));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_1D_vector_non_contiguous_dim1_add5_pack_axis_1_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C_coefficient = 8;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::make_shape(N, (C) +7, H, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C / P, H, W));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<1>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)(C / P), (int64_t)P, (int64_t)H, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 3, 4, 2};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C / P, H, W));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_1D_vector_non_contiguous_dim1_add5_pack_axis_0_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N_coefficient = 2;
    constexpr size_t N = N_coefficient * P;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::make_shape(N, (C) +7, H, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N / P, C, H, W));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<0>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)(N / P), (int64_t)P, (int64_t)C, (int64_t)H, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 2, 3, 4, 1};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N / P, C, H, W));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_2D_vector_contiguous_pack_axis_0_1_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N_coefficient = 2;
    constexpr size_t N = N_coefficient * P;
    constexpr size_t C_coefficient = 8;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    alignas(32) auto ntt_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N / P, C / P, H, W));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<0, 1>);

    // ORT reference implementation
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t reshape_data[] = {(int64_t)(N / P), (int64_t)P, (int64_t)(C / P), (int64_t)P, (int64_t)H, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 2, 4, 5, 1, 3};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N / P, C / P, H, W));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_2D_vector_contiguous_pack_axis_1_2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C_coefficient = 8;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H_coefficient = 4;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    alignas(32) auto ntt_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N, C / P, H / P, W));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<1, 2>);

    // ORT reference implementation
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)(C / P), (int64_t)P, (int64_t)(H / P), (int64_t)P, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 3, 5, 2, 4};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N, C / P, H / P, W));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_2D_vector_contiguous_pack_axis_2_3_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H_coefficient = 4;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W_coefficient = 4;
    constexpr size_t W = W_coefficient * P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    alignas(32) auto ntt_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N, C, H / P, W / P));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<2, 3>);

    // ORT reference implementation
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)(H / P), (int64_t)P, (int64_t)(W / P), (int64_t)P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 4, 3, 5};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N, C, H / P, W / P));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_2D_vector_non_contiguous_dim2_add5_pack_axis_0_1_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N_coefficient = 2;
    constexpr size_t N = N_coefficient * P;
    constexpr size_t C_coefficient = 8;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 2)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::make_shape(N, C, (H) +7, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N / P, C / P, H, W));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<0, 1>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)(N / P), (int64_t)P, (int64_t)(C / P), (int64_t)P, (int64_t)H, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 2, 4, 5, 1, 3};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N / P, C / P, H, W));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_2D_vector_non_contiguous_dim2_add5_pack_axis_1_2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C_coefficient = 8;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H_coefficient = 4;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 2)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::make_shape(N, C, (H) +7, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N, C / P, H / P, W));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<1, 2>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)(C / P), (int64_t)P, (int64_t)(H / P), (int64_t)P, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 3, 5, 2, 4};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N, C / P, H / P, W));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_2D_vector_non_contiguous_dim2_add5_pack_axis_2_3_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H_coefficient = 4;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W_coefficient = 4;
    constexpr size_t W = W_coefficient * P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 2)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::make_shape(N, C, (H) +7, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N, C, H / P, W / P));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<2, 3>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)(H / P), (int64_t)P, (int64_t)(W / P), (int64_t)P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 4, 3, 5};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N, C, H / P, W / P));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_2D_vector_non_contiguous_dim2_mul2_pack_axis_0_1_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N_coefficient = 2;
    constexpr size_t N = N_coefficient * P;
    constexpr size_t C_coefficient = 8;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 2)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::make_shape(N, C, (H) *2, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N / P, C / P, H, W));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<0, 1>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)(N / P), (int64_t)P, (int64_t)(C / P), (int64_t)P, (int64_t)H, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 2, 4, 5, 1, 3};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N / P, C / P, H, W));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_2D_vector_non_contiguous_dim2_mul2_pack_axis_1_2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C_coefficient = 8;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H_coefficient = 4;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 2)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::make_shape(N, C, (H) *2, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N, C / P, H / P, W));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<1, 2>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)(C / P), (int64_t)P, (int64_t)(H / P), (int64_t)P, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 3, 5, 2, 4};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N, C / P, H / P, W));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_2D_vector_non_contiguous_dim2_mul2_pack_axis_2_3_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H_coefficient = 4;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W_coefficient = 4;
    constexpr size_t W = W_coefficient * P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 2)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::make_shape(N, C, (H) *2, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N, C, H / P, W / P));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<2, 3>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)(H / P), (int64_t)P, (int64_t)(W / P), (int64_t)P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 4, 3, 5};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N, C, H / P, W / P));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_2D_vector_non_contiguous_dim1_mul2_pack_axis_0_1_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N_coefficient = 2;
    constexpr size_t N = N_coefficient * P;
    constexpr size_t C_coefficient = 8;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::make_shape(N, (C) *2, H, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N / P, C / P, H, W));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<0, 1>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)(N / P), (int64_t)P, (int64_t)(C / P), (int64_t)P, (int64_t)H, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 2, 4, 5, 1, 3};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N / P, C / P, H, W));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_2D_vector_non_contiguous_dim1_mul2_pack_axis_1_2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C_coefficient = 8;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H_coefficient = 4;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::make_shape(N, (C) *2, H, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N, C / P, H / P, W));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<1, 2>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)(C / P), (int64_t)P, (int64_t)(H / P), (int64_t)P, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 3, 5, 2, 4};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N, C / P, H / P, W));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_2D_vector_non_contiguous_dim1_mul2_pack_axis_2_3_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H_coefficient = 4;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W_coefficient = 4;
    constexpr size_t W = W_coefficient * P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::make_shape(N, (C) *2, H, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N, C, H / P, W / P));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<2, 3>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)(H / P), (int64_t)P, (int64_t)(W / P), (int64_t)P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 4, 3, 5};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N, C, H / P, W / P));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_2D_vector_non_contiguous_dim1_add5_pack_axis_0_1_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N_coefficient = 2;
    constexpr size_t N = N_coefficient * P;
    constexpr size_t C_coefficient = 8;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::make_shape(N, (C) +7, H, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N / P, C / P, H, W));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<0, 1>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)(N / P), (int64_t)P, (int64_t)(C / P), (int64_t)P, (int64_t)H, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 2, 4, 5, 1, 3};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N / P, C / P, H, W));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_2D_vector_non_contiguous_dim1_add5_pack_axis_1_2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C_coefficient = 8;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H_coefficient = 4;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W = 4;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::make_shape(N, (C) +7, H, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N, C / P, H / P, W));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<1, 2>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)(C / P), (int64_t)P, (int64_t)(H / P), (int64_t)P, (int64_t)W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 3, 5, 2, 4};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N, C / P, H / P, W));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_2D_vector_non_contiguous_dim1_add5_pack_axis_2_3_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H_coefficient = 4;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W_coefficient = 4;
    constexpr size_t W = W_coefficient * P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::make_shape(N, (C) +7, H, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N, C, H / P, W / P));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<2, 3>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    continuous_input(n, c, h, w) = ntt_input(n, c, h, w);
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)(H / P), (int64_t)P, (int64_t)(W / P), (int64_t)P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 4, 3, 5};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N, C, H / P, W / P));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_1D_vector_contiguous_pack_axis_4_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D_coefficient = 2;
    constexpr size_t D = D_coefficient * P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    alignas(32) auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W, D>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C, H, W, D / P>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<4>);

    // ORT reference implementation
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)H, (int64_t)W, (int64_t)(D / P), (int64_t)P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 3, 4, 5};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C, H, W, D / P>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_1D_vector_contiguous_pack_axis_3_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W_coefficient = 4;
    constexpr size_t W = W_coefficient * P;
    constexpr size_t D = 2;
    float min_input = -10.0f;
    float max_input = 10.0f;

    alignas(32) auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W, D>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C, H, W / P, D>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    // ORT reference implementation
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)H, (int64_t)(W / P), (int64_t)P, (int64_t)D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 3, 5, 4};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C, H, W / P, D>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_1D_vector_contiguous_pack_axis_2_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H_coefficient = 4;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float min_input = -10.0f;
    float max_input = 10.0f;

    alignas(32) auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W, D>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C, H / P, W, D>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // ORT reference implementation
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)(H / P), (int64_t)P, (int64_t)W, (int64_t)D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 4, 5, 3};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C, H / P, W, D>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_1D_vector_contiguous_pack_axis_1_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C_coefficient = 8;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float min_input = -10.0f;
    float max_input = 10.0f;

    alignas(32) auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W, D>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C / P, H, W, D>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<1>);

    // ORT reference implementation
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)(C / P), (int64_t)P, (int64_t)H, (int64_t)W, (int64_t)D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 3, 4, 5, 2};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C / P, H, W, D>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_1D_vector_contiguous_pack_axis_0_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N_coefficient = 2;
    constexpr size_t N = N_coefficient * P;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float min_input = -10.0f;
    float max_input = 10.0f;

    alignas(32) auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W, D>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N / P, C, H, W, D>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<0>);

    // ORT reference implementation
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t reshape_data[] = {(int64_t)(N / P), (int64_t)P, (int64_t)C, (int64_t)H, (int64_t)W, (int64_t)D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 2, 3, 4, 5, 1};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N / P, C, H, W, D>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_1D_vector_non_contiguous_dim1_mul2_pack_axis_4_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D_coefficient = 2;
    constexpr size_t D = D_coefficient * P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::fixed_shape_v<N, (C) *2, H, W, D>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W, D>,
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C, H, W, D / P>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<4>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W, D>);
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    for (size_t d = 0; d < D; d++) {
                        continuous_input(n, c, h, w, d) = ntt_input(n, c, h, w, d);
                    }
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)H, (int64_t)W, (int64_t)(D / P), (int64_t)P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 3, 4, 5};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C, H, W, D / P>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_1D_vector_non_contiguous_dim1_mul2_pack_axis_3_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W_coefficient = 4;
    constexpr size_t W = W_coefficient * P;
    constexpr size_t D = 2;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::fixed_shape_v<N, (C) *2, H, W, D>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W, D>,
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C, H, W / P, D>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W, D>);
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    for (size_t d = 0; d < D; d++) {
                        continuous_input(n, c, h, w, d) = ntt_input(n, c, h, w, d);
                    }
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)H, (int64_t)(W / P), (int64_t)P, (int64_t)D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 3, 5, 4};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C, H, W / P, D>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_1D_vector_non_contiguous_dim1_mul2_pack_axis_2_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H_coefficient = 4;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::fixed_shape_v<N, (C) *2, H, W, D>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W, D>,
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C, H / P, W, D>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W, D>);
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    for (size_t d = 0; d < D; d++) {
                        continuous_input(n, c, h, w, d) = ntt_input(n, c, h, w, d);
                    }
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)(H / P), (int64_t)P, (int64_t)W, (int64_t)D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 4, 5, 3};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C, H / P, W, D>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_1D_vector_non_contiguous_dim1_mul2_pack_axis_1_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C_coefficient = 8;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::fixed_shape_v<N, (C) *2, H, W, D>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W, D>,
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C / P, H, W, D>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<1>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W, D>);
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    for (size_t d = 0; d < D; d++) {
                        continuous_input(n, c, h, w, d) = ntt_input(n, c, h, w, d);
                    }
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)(C / P), (int64_t)P, (int64_t)H, (int64_t)W, (int64_t)D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 3, 4, 5, 2};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N, C / P, H, W, D>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_1D_vector_non_contiguous_dim1_mul2_pack_axis_0_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N_coefficient = 2;
    constexpr size_t N = N_coefficient * P;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::fixed_shape_v<N, (C) *2, H, W, D>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W, D>,
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N / P, C, H, W, D>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<0>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W, D>);
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    for (size_t d = 0; d < D; d++) {
                        continuous_input(n, c, h, w, d) = ntt_input(n, c, h, w, d);
                    }
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)(N / P), (int64_t)P, (int64_t)C, (int64_t)H, (int64_t)W, (int64_t)D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 2, 3, 4, 5, 1};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N / P, C, H, W, D>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_2D_vector_contiguous_pack_axis_0_1_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N_coefficient = 2;
    constexpr size_t N = N_coefficient * P;
    constexpr size_t C_coefficient = 8;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float min_input = -10.0f;
    float max_input = 10.0f;

    alignas(32) auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W, D>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N / P, C / P, H, W, D>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<0, 1>);

    // ORT reference implementation
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t reshape_data[] = {(int64_t)(N / P), (int64_t)P, (int64_t)(C / P), (int64_t)P, (int64_t)H, (int64_t)W, (int64_t)D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 2, 4, 5, 6, 1, 3};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N / P, C / P, H, W, D>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_2D_vector_contiguous_pack_axis_1_2_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C_coefficient = 8;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H_coefficient = 4;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float min_input = -10.0f;
    float max_input = 10.0f;

    alignas(32) auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W, D>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N, C / P, H / P, W, D>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<1, 2>);

    // ORT reference implementation
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)(C / P), (int64_t)P, (int64_t)(H / P), (int64_t)P, (int64_t)W, (int64_t)D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 3, 5, 6, 2, 4};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N, C / P, H / P, W, D>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_2D_vector_contiguous_pack_axis_2_3_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H_coefficient = 4;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W_coefficient = 4;
    constexpr size_t W = W_coefficient * P;
    constexpr size_t D = 2;
    float min_input = -10.0f;
    float max_input = 10.0f;

    alignas(32) auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W, D>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N, C, H / P, W / P, D>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<2, 3>);

    // ORT reference implementation
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)(H / P), (int64_t)P, (int64_t)(W / P), (int64_t)P, (int64_t)D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 4, 6, 3, 5};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N, C, H / P, W / P, D>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_2D_vector_contiguous_pack_axis_3_4_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W_coefficient = 4;
    constexpr size_t W = W_coefficient * P;
    constexpr size_t D_coefficient = 2;
    constexpr size_t D = D_coefficient * P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    alignas(32) auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W, D>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N, C, H, W / P, D / P>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<3, 4>);

    // ORT reference implementation
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)H, (int64_t)(W / P), (int64_t)P, (int64_t)(D / P), (int64_t)P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 3, 5, 4, 6};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N, C, H, W / P, D / P>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_2D_vector_non_contiguous_dim1_mul2_pack_axis_0_1_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N_coefficient = 2;
    constexpr size_t N = N_coefficient * P;
    constexpr size_t C_coefficient = 8;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::fixed_shape_v<N, (C) *2, H, W, D>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W, D>,
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N / P, C / P, H, W, D>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<0, 1>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W, D>);
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    for (size_t d = 0; d < D; d++) {
                        continuous_input(n, c, h, w, d) = ntt_input(n, c, h, w, d);
                    }
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)(N / P), (int64_t)P, (int64_t)(C / P), (int64_t)P, (int64_t)H, (int64_t)W, (int64_t)D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 2, 4, 5, 6, 1, 3};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N / P, C / P, H, W, D>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_2D_vector_non_contiguous_dim1_mul2_pack_axis_1_2_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C_coefficient = 8;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H_coefficient = 4;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::fixed_shape_v<N, (C) *2, H, W, D>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W, D>,
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N, C / P, H / P, W, D>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<1, 2>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W, D>);
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    for (size_t d = 0; d < D; d++) {
                        continuous_input(n, c, h, w, d) = ntt_input(n, c, h, w, d);
                    }
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)(C / P), (int64_t)P, (int64_t)(H / P), (int64_t)P, (int64_t)W, (int64_t)D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 3, 5, 6, 2, 4};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N, C / P, H / P, W, D>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_2D_vector_non_contiguous_dim1_mul2_pack_axis_2_3_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H_coefficient = 4;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W_coefficient = 4;
    constexpr size_t W = W_coefficient * P;
    constexpr size_t D = 2;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::fixed_shape_v<N, (C) *2, H, W, D>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W, D>,
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N, C, H / P, W / P, D>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<2, 3>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W, D>);
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    for (size_t d = 0; d < D; d++) {
                        continuous_input(n, c, h, w, d) = ntt_input(n, c, h, w, d);
                    }
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)(H / P), (int64_t)P, (int64_t)(W / P), (int64_t)P, (int64_t)D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 4, 6, 3, 5};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N, C, H / P, W / P, D>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_2D_vector_non_contiguous_dim1_mul2_pack_axis_3_4_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W_coefficient = 4;
    constexpr size_t W = W_coefficient * P;
    constexpr size_t D_coefficient = 2;
    constexpr size_t D = D_coefficient * P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::fixed_shape_v<N, (C) *2, H, W, D>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W, D>,
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N, C, H, W / P, D / P>);

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<3, 4>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W, D>);
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    for (size_t d = 0; d < D; d++) {
                        continuous_input(n, c, h, w, d) = ntt_input(n, c, h, w, d);
                    }
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)H, (int64_t)(W / P), (int64_t)P, (int64_t)(D / P), (int64_t)P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 3, 5, 4, 6};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::fixed_shape_v<N, C, H, W / P, D / P>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_1D_vector_contiguous_pack_axis_4_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D_coefficient = 2;
    constexpr size_t D = D_coefficient * P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    alignas(32) auto ntt_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W, D));
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C, H, W, D / P));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<4>);

    // ORT reference implementation
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)H, (int64_t)W, (int64_t)(D / P), (int64_t)P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 3, 4, 5};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C, H, W, D / P));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_1D_vector_contiguous_pack_axis_3_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W_coefficient = 4;
    constexpr size_t W = W_coefficient * P;
    constexpr size_t D = 2;
    float min_input = -10.0f;
    float max_input = 10.0f;

    alignas(32) auto ntt_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W, D));
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C, H, W / P, D));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    // ORT reference implementation
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)H, (int64_t)(W / P), (int64_t)P, (int64_t)D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 3, 5, 4};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C, H, W / P, D));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_1D_vector_contiguous_pack_axis_2_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H_coefficient = 4;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float min_input = -10.0f;
    float max_input = 10.0f;

    alignas(32) auto ntt_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W, D));
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C, H / P, W, D));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // ORT reference implementation
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)(H / P), (int64_t)P, (int64_t)W, (int64_t)D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 4, 5, 3};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C, H / P, W, D));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_1D_vector_contiguous_pack_axis_1_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C_coefficient = 8;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float min_input = -10.0f;
    float max_input = 10.0f;

    alignas(32) auto ntt_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W, D));
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C / P, H, W, D));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<1>);

    // ORT reference implementation
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)(C / P), (int64_t)P, (int64_t)H, (int64_t)W, (int64_t)D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 3, 4, 5, 2};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C / P, H, W, D));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_1D_vector_contiguous_pack_axis_0_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N_coefficient = 2;
    constexpr size_t N = N_coefficient * P;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float min_input = -10.0f;
    float max_input = 10.0f;

    alignas(32) auto ntt_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W, D));
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N / P, C, H, W, D));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<0>);

    // ORT reference implementation
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t reshape_data[] = {(int64_t)(N / P), (int64_t)P, (int64_t)C, (int64_t)H, (int64_t)W, (int64_t)D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 2, 3, 4, 5, 1};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N / P, C, H, W, D));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_1D_vector_non_contiguous_dim1_mul2_pack_axis_4_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D_coefficient = 2;
    constexpr size_t D = D_coefficient * P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::make_shape(N, (C) *2, H, W, D));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W, D),
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C, H, W, D / P));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<4>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W, D));
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    for (size_t d = 0; d < D; d++) {
                        continuous_input(n, c, h, w, d) = ntt_input(n, c, h, w, d);
                    }
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)H, (int64_t)W, (int64_t)(D / P), (int64_t)P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 3, 4, 5};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C, H, W, D / P));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_1D_vector_non_contiguous_dim1_mul2_pack_axis_3_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W_coefficient = 4;
    constexpr size_t W = W_coefficient * P;
    constexpr size_t D = 2;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::make_shape(N, (C) *2, H, W, D));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W, D),
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C, H, W / P, D));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W, D));
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    for (size_t d = 0; d < D; d++) {
                        continuous_input(n, c, h, w, d) = ntt_input(n, c, h, w, d);
                    }
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)H, (int64_t)(W / P), (int64_t)P, (int64_t)D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 3, 5, 4};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C, H, W / P, D));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_1D_vector_non_contiguous_dim1_mul2_pack_axis_2_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H_coefficient = 4;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::make_shape(N, (C) *2, H, W, D));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W, D),
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C, H / P, W, D));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W, D));
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    for (size_t d = 0; d < D; d++) {
                        continuous_input(n, c, h, w, d) = ntt_input(n, c, h, w, d);
                    }
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)(H / P), (int64_t)P, (int64_t)W, (int64_t)D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 4, 5, 3};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C, H / P, W, D));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_1D_vector_non_contiguous_dim1_mul2_pack_axis_1_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C_coefficient = 8;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::make_shape(N, (C) *2, H, W, D));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W, D),
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C / P, H, W, D));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<1>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W, D));
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    for (size_t d = 0; d < D; d++) {
                        continuous_input(n, c, h, w, d) = ntt_input(n, c, h, w, d);
                    }
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)(C / P), (int64_t)P, (int64_t)H, (int64_t)W, (int64_t)D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 3, 4, 5, 2};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N, C / P, H, W, D));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_1D_vector_non_contiguous_dim1_mul2_pack_axis_0_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N_coefficient = 2;
    constexpr size_t N = N_coefficient * P;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::make_shape(N, (C) *2, H, W, D));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W, D),
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N / P, C, H, W, D));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<0>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W, D));
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    for (size_t d = 0; d < D; d++) {
                        continuous_input(n, c, h, w, d) = ntt_input(n, c, h, w, d);
                    }
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)(N / P), (int64_t)P, (int64_t)C, (int64_t)H, (int64_t)W, (int64_t)D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 2, 3, 4, 5, 1};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(N / P, C, H, W, D));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_2D_vector_contiguous_pack_axis_0_1_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N_coefficient = 2;
    constexpr size_t N = N_coefficient * P;
    constexpr size_t C_coefficient = 8;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float min_input = -10.0f;
    float max_input = 10.0f;

    alignas(32) auto ntt_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W, D));
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N / P, C / P, H, W, D));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<0, 1>);

    // ORT reference implementation
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t reshape_data[] = {(int64_t)(N / P), (int64_t)P, (int64_t)(C / P), (int64_t)P, (int64_t)H, (int64_t)W, (int64_t)D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 2, 4, 5, 6, 1, 3};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N / P, C / P, H, W, D));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_2D_vector_contiguous_pack_axis_1_2_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C_coefficient = 8;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H_coefficient = 4;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float min_input = -10.0f;
    float max_input = 10.0f;

    alignas(32) auto ntt_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W, D));
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N, C / P, H / P, W, D));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<1, 2>);

    // ORT reference implementation
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)(C / P), (int64_t)P, (int64_t)(H / P), (int64_t)P, (int64_t)W, (int64_t)D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 3, 5, 6, 2, 4};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N, C / P, H / P, W, D));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_2D_vector_contiguous_pack_axis_2_3_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H_coefficient = 4;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W_coefficient = 4;
    constexpr size_t W = W_coefficient * P;
    constexpr size_t D = 2;
    float min_input = -10.0f;
    float max_input = 10.0f;

    alignas(32) auto ntt_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W, D));
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N, C, H / P, W / P, D));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<2, 3>);

    // ORT reference implementation
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)(H / P), (int64_t)P, (int64_t)(W / P), (int64_t)P, (int64_t)D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 4, 6, 3, 5};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N, C, H / P, W / P, D));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_2D_vector_contiguous_pack_axis_3_4_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W_coefficient = 4;
    constexpr size_t W = W_coefficient * P;
    constexpr size_t D_coefficient = 2;
    constexpr size_t D = D_coefficient * P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    alignas(32) auto ntt_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W, D));
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N, C, H, W / P, D / P));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<3, 4>);

    // ORT reference implementation
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)H, (int64_t)(W / P), (int64_t)P, (int64_t)(D / P), (int64_t)P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 3, 5, 4, 6};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N, C, H, W / P, D / P));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_2D_vector_non_contiguous_dim1_mul2_pack_axis_0_1_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N_coefficient = 2;
    constexpr size_t N = N_coefficient * P;
    constexpr size_t C_coefficient = 8;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::make_shape(N, (C) *2, H, W, D));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W, D),
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N / P, C / P, H, W, D));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<0, 1>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W, D));
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    for (size_t d = 0; d < D; d++) {
                        continuous_input(n, c, h, w, d) = ntt_input(n, c, h, w, d);
                    }
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)(N / P), (int64_t)P, (int64_t)(C / P), (int64_t)P, (int64_t)H, (int64_t)W, (int64_t)D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 2, 4, 5, 6, 1, 3};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N / P, C / P, H, W, D));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_2D_vector_non_contiguous_dim1_mul2_pack_axis_1_2_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C_coefficient = 8;
    constexpr size_t C = C_coefficient * P;
    constexpr size_t H_coefficient = 4;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::make_shape(N, (C) *2, H, W, D));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W, D),
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N, C / P, H / P, W, D));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<1, 2>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W, D));
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    for (size_t d = 0; d < D; d++) {
                        continuous_input(n, c, h, w, d) = ntt_input(n, c, h, w, d);
                    }
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)(C / P), (int64_t)P, (int64_t)(H / P), (int64_t)P, (int64_t)W, (int64_t)D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 3, 5, 6, 2, 4};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N, C / P, H / P, W, D));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_2D_vector_non_contiguous_dim1_mul2_pack_axis_2_3_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H_coefficient = 4;
    constexpr size_t H = H_coefficient * P;
    constexpr size_t W_coefficient = 4;
    constexpr size_t W = W_coefficient * P;
    constexpr size_t D = 2;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::make_shape(N, (C) *2, H, W, D));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W, D),
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N, C, H / P, W / P, D));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<2, 3>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W, D));
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    for (size_t d = 0; d < D; d++) {
                        continuous_input(n, c, h, w, d) = ntt_input(n, c, h, w, d);
                    }
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)(H / P), (int64_t)P, (int64_t)(W / P), (int64_t)P, (int64_t)D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 4, 6, 3, 5};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N, C, H / P, W / P, D));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, dynamic_2D_vector_non_contiguous_dim1_mul2_pack_axis_3_4_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W_coefficient = 4;
    constexpr size_t W = W_coefficient * P;
    constexpr size_t D_coefficient = 2;
    constexpr size_t D = D_coefficient * P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // Create non-contiguous tensor (on dimension 1)
    alignas(32) auto big_tensor = ntt::make_tensor<float>(ntt::make_shape(N, (C) *2, H, W, D));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W, D),
        big_tensor.strides());

    // Create output tensor
    alignas(32) auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N, C, H, W / P, D / P));

    // Execute pack operation
    ntt::pack(ntt_input, ntt_output1, ntt::fixed_shape_v<3, 4>);

    // Copy to contiguous tensor for ORT reference
    alignas(32) auto continuous_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W, D));
    
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    for (size_t d = 0; d < D; d++) {
                        continuous_input(n, c, h, w, d) = ntt_input(n, c, h, w, d);
                    }
                }
            }
        }
    }

    // ORT reference implementation
        auto ort_input = NttTest::ntt2ort(continuous_input);
    int64_t reshape_data[] = {(int64_t)N, (int64_t)C, (int64_t)H, (int64_t)(W / P), (int64_t)P, (int64_t)(D / P), (int64_t)P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);
    
    int64_t perms[] = {0, 1, 2, 3, 5, 4, 6};
    auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));

    // Compare results
    alignas(32) auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P, P>>(ntt::make_shape(N, C, H, W / P, D / P));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
