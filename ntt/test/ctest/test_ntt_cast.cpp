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
#include "ntt_test.h"
#include "ortki_helper.h"
#include <gtest/gtest.h>
#include <nncase/bfloat16.h>
#include <nncase/float8.h>
#include <nncase/half.h>
#include <nncase/ntt/ntt.h>
#include <ortki/operators.h>

using namespace nncase;
using namespace ortki;

TEST(CastTest_fp32_i32, NoVectorize) {
    auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, -32761.0f, 32761.0f, true, false);
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int32_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output);


    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT32);

    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int32_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}

TEST(CastTest_fp32_i32, 1D_Vectorize) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, -32761.0f, 32761.0f, true, false);
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int32_t, P>>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output);


    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT32);

    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int32_t, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}

TEST(CastTest_fp32_i32, 2D_Vectorize) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, -32761.0f, 32761.0f, true, false);
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int32_t, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output);


    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT32);

    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int32_t, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}


TEST(CastTest_fp32_i8, NoVectorize) {
    auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, -11.0f, 11.0f, true, false);
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int8_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output);


    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT8);

    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int8_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}

TEST(CastTest_fp32_i8, Vectorize) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, -11.0f, 11.0f, true, false);
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int8_t, 4 * P>>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output);

    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 4, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);

    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));

    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_INT8);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int8_t, 4 * P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}


TEST(CastTest_i8_fp32, NoVectorize) {
    auto ntt_input = ntt::make_tensor<int8_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, -11, 11, true, false);
    // Create output tensor
    auto ntt_output = ntt::make_tensor<float>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output);


    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT);

    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<float>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}

TEST(CastTest_i8_fp32, Vectorize) {
    constexpr size_t P = NTT_VLEN / (sizeof(int8_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<int8_t, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, -11, 11, true, false);
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<float, 4, P / 4>>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output);


    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 4, P / 4};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);

    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));

    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_FLOAT);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<float, 4, P / 4>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}

TEST(CastTest_fp32_b, NoVectorize) {
    auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, -11.0f, 11.0f, true, false);
    // Create output tensor
    auto ntt_output = ntt::make_tensor<bool>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output);

    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BOOL);

    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<bool>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}

TEST(CastTest_fp32_b, 1D_Vectorize) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, -11.0f, 11.0f, true, false);
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<bool, P>>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output);

    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 1, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);

    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));

    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_BOOL);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bool, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}

TEST(CastTest_fp32_b, 2D_Vectorize) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, -11.0f, 11.0f, true, false);
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<bool, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output);

    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 4, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);

    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));

    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_BOOL);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bool, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}

TEST(CastTest_b_fp32, NoVectorize) {
    auto ntt_input = ntt::make_tensor<bool>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, -11, 11, true, false);
    // Create output tensor
    auto ntt_output = ntt::make_tensor<float>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output);


    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT);

    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<float>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}

TEST(CastTest_b_fp32, 1D_Vectorize) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<bool, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, -11, 11, true, false);
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output);


    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);

    int64_t perms_data[] = {0, 1, 2, 3};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));

    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_FLOAT);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}


TEST(CastTest_b_fp32, 2D_Vectorize) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<bool, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, -11, 11, true, false);
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<float, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output);


    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 4, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);

    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));

    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_FLOAT);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<float, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}

TEST(CastTest_bf16_fp8e4m3, fixed_2D_vector_contiguous_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(bfloat16) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<bfloat16, 2, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, -10.0_bf16, 10.0_bf16, true, false);
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<float_e4m3_t, 2 * P>>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output);


#if 1
    // auto ntt_input2 = ntt::make_tensor<ntt::vector<bfloat16, 2, P>>(ntt::fixed_shape_v<8, 80, 8>);
    // ntt::cast(ntt_output, ntt_input2);
    // EXPECT_TRUE(NttTest::compare_tensor(ntt_input, ntt_input2));
    auto ntt_golden = ntt::make_tensor<ntt::vector<float_e4m3_t, 2 * P>>(ntt::fixed_shape_v<8, 80, 8>);

    ntt::apply(ntt_input.shape(), [&](auto& index){
      ntt::apply(ntt_input(index).shape(), [&](auto& sub_index){
        auto val = static_cast<float_e4m3_t>(ntt_input(index)(sub_index));
        size_t linear_idx = linear_offset(sub_index, ntt_input(index).shape());
        (ntt_golden)(index)(linear_idx) = val;
      });
    });
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));

#else
    auto ntt_scalar_input = ntt::make_tensor<bfloat16>(ntt::fixed_shape_v<8, 80 * 2, 8 * P>);
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<1, 2>);
    auto ntt_golden_scalar = ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });

    // auto ntt_scalar_reshape = ntt::make_tensor<float_e4m3_t>(ntt::fixed_shape_v<8, 80, 2 * 8 * P>);
    // ntt::reshape(ntt_golden_scalar, ntt_scalar_reshape);

    auto ntt_golden_vector = ntt::make_tensor<ntt::vector<float_e4m3_t, 2, P>>(ntt::fixed_shape_v<8, 80, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<1, 2>);


    auto ntt_golden = ntt::make_tensor<ntt::vector<float_e4m3_t, 2 * P>>(ntt::fixed_shape_v<8, 80, 8>);
    ntt::apply(ntt_golden.shape(), [&](auto& index){
    //   (ntt_golden)(index) = static_cast<ntt::vector<float_e4m3_t, 2 * P>>(ntt_golden_vector(index));
      ntt::reshape(ntt_golden_vector(index), (ntt_golden)(index) );
    });

    // auto& ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
#endif
}

TEST(CastTest_fp8e4m3_fp16, fixed_1D_vector_contiguous_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e4m3_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e4m3_t, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, float_e4m3_t(-16.0f), float_e4m3_t(16.0f), true, false);
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<half, 2, P / 2>>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output);


    auto ntt_golden = ntt::make_tensor<ntt::vector<half, 2, P / 2>>(ntt::fixed_shape_v<8, 80, 8>);
    ntt::apply((ntt_input).shape(), [&](auto& index1){
        auto &v_in = (ntt_input)(index1);
        auto &v_out = ntt_golden(index1);
        ntt::apply(v_in.shape(), [&](auto& index2) {
            auto val = static_cast<half>(v_in(index2));
            auto offset = linear_offset(index2, v_in.shape());
            auto index3 = unravel_index(offset, v_out.shape());
            v_out(index3) = val;
        });
    });

    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}