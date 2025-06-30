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
#include "ntt_test.h"
#include "ortki_helper.h"
#include <gtest/gtest.h>
#include <iostream>
#include <nncase/ntt/ntt.h>
#include <ortki/operators.h>

using namespace nncase;
using namespace ortki;

#define RUN_ORT_SLICE(STARTS, STOPS, AXES, STEPS)                              \
    auto ort_input = NttTest::ntt2ort(ntt_input);                              \
    int64_t starts_buf[] = STARTS;                                             \
    int64_t stops_buf[] = STOPS;                                               \
    int64_t axes_buf[] = AXES;                                                 \
    int64_t steps_buf[] = STEPS;                                               \
    int64_t shape[] = {std::size(starts_buf)};                                 \
    auto starts = make_tensor(reinterpret_cast<void *>(starts_buf),            \
                              DataType_INT64, shape, 1);                       \
    auto stops = make_tensor(reinterpret_cast<void *>(stops_buf),              \
                             DataType_INT64, shape, 1);                        \
    auto axes = make_tensor(reinterpret_cast<void *>(axes_buf),                \
                            DataType_INT64, shape, 1);                         \
    auto steps = make_tensor(reinterpret_cast<void *>(steps_buf),              \
                             DataType_INT64, shape, 1);                        \
    auto ort_output = ortki_Slice(ort_input, starts, stops, axes, steps);      \
    NttTest::ort2ntt(ort_output, ntt_output2);                                 \
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));

TEST(SliceTestFloat, NoPack_dim_1_step_eq_1) {
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<32, 32>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<32, 4>);
    ntt::slice(ntt_input, ntt_output1, fixed_shape_v<4>, fixed_shape_v<8>,
               ntt::fixed_shape_v<1>, ntt::fixed_shape_v<1>);

    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<32, 4>);

    // ort
    RUN_ORT_SLICE({4}, {8}, {1}, {1})
}

TEST(SliceTestFloat, NoPack_dim_1_step_gt_1) {
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<32, 32>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<32, 4>);
    ntt::slice(ntt_input, ntt_output1, fixed_shape_v<4>, fixed_shape_v<12>,
               ntt::fixed_shape_v<1>, ntt::fixed_shape_v<2>);

    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<32, 4>);
    // ort
    RUN_ORT_SLICE({4}, {12}, {1}, {2})
}

TEST(SliceTestFloat, NoPack_dim_0_step_eq_1) {
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<32, 32>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<4, 32>);
    ntt::slice(ntt_input, ntt_output1, fixed_shape_v<4>, fixed_shape_v<8>,
               ntt::fixed_shape_v<0>, ntt::fixed_shape_v<1>);

    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<4, 32>);
    // ort
    RUN_ORT_SLICE({4}, {8}, {0}, {1})
}

TEST(SliceTestFloat, NoPack_dim_0_step_gt_1) {
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<32, 32>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<4, 32>);
    ntt::slice(ntt_input, ntt_output1, fixed_shape_v<4>, fixed_shape_v<12>,
               ntt::fixed_shape_v<0>, ntt::fixed_shape_v<2>);

    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<4, 32>);
    // ort
    RUN_ORT_SLICE({4}, {12}, {0}, {2})
}

#define RUN_ORT_SLICE2(STARTS, STOPS, AXES, STEPS, LEN)                        \
    auto ort_input = NttTest::ntt2ort(ntt_input);                              \
    int64_t shape[] = {LEN};                                                   \
    auto starts = make_tensor(reinterpret_cast<void *>(STARTS),                \
                              DataType_INT64, shape, 1);                       \
    auto stops = make_tensor(reinterpret_cast<void *>(STOPS), DataType_INT64,  \
                             shape, 1);                                        \
    auto axes =                                                                \
        make_tensor(reinterpret_cast<void *>(AXES), DataType_INT64, shape, 1); \
    auto steps = make_tensor(reinterpret_cast<void *>(STEPS), DataType_INT64,  \
                             shape, 1);                                        \
    auto ort_output = ortki_Slice(ort_input, starts, stops, axes, steps);      \
    NttTest::ort2ntt(ort_output, ntt_output2);                                 \
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));

TEST(SliceTestFloat, NoPack_dim_0_1_step_eq_1) {
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<32, 32>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<4, 4>);
    ntt::slice(ntt_input, ntt_output1, fixed_shape_v<1, 4>, fixed_shape_v<5, 8>,
               ntt::fixed_shape_v<0, 1>, ntt::fixed_shape_v<1, 1>);

    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<4, 4>);
    // ort
    int64_t starts_buf[] = {1, 4};
    int64_t stops_buf[] = {5, 8};
    int64_t axes_buf[] = {0, 1};
    int64_t steps_buf[] = {1, 1};
    RUN_ORT_SLICE2(starts_buf, stops_buf, axes_buf, steps_buf, 2)
}

TEST(SliceTestFloat, NoPack_dim_0_1_step_gt_1) {
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<32, 32>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<4, 4>);
    ntt::slice(ntt_input, ntt_output1, fixed_shape_v<1, 4>,
               fixed_shape_v<8, 12>, ntt::fixed_shape_v<0, 1>,
               ntt::fixed_shape_v<2, 2>);

    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<4, 4>);
    // ort
    int64_t starts_buf[] = {1, 4};
    int64_t stops_buf[] = {8, 12};
    int64_t axes_buf[] = {0, 1};
    int64_t steps_buf[] = {2, 2};
    RUN_ORT_SLICE2(starts_buf, stops_buf, axes_buf, steps_buf, 2)
}

TEST(SliceTestFloat, NoPack_dim_multiple_step_eq_1) {
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<1, 3, 32, 32>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<1, 2, 4, 4>);
    ntt::slice(ntt_input, ntt_output1, fixed_shape_v<1, 1, 4>,
               fixed_shape_v<3, 5, 8>, ntt::fixed_shape_v<1, 2, 3>,
               ntt::fixed_shape_v<1, 1, 1>);

    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<1, 2, 4, 4>);
    // ort
    int64_t starts_buf[] = {1, 1, 4};
    int64_t stops_buf[] = {3, 5, 8};
    int64_t axes_buf[] = {1, 2, 3};
    int64_t steps_buf[] = {1, 1, 1};
    RUN_ORT_SLICE2(starts_buf, stops_buf, axes_buf, steps_buf, 3)
}

TEST(SliceTestFloat, NoPack_dim_multiple_step_gt_1) {
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<1, 3, 32, 32>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<1, 2, 5, 6>);
    ntt::slice(ntt_input, ntt_output1, fixed_shape_v<0, 1, 4>,
               fixed_shape_v<3, 10, 16>, ntt::fixed_shape_v<1, 2, 3>,
               ntt::fixed_shape_v<2, 2, 2>);

    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<1, 2, 5, 6>);
    // ort
    int64_t starts_buf[] = {0, 1, 4};
    int64_t stops_buf[] = {3, 10, 16};
    int64_t axes_buf[] = {1, 2, 3};
    int64_t steps_buf[] = {2, 2, 2};
    RUN_ORT_SLICE2(starts_buf, stops_buf, axes_buf, steps_buf, 3)
}

TEST(SliceTestFloat, Pack_fixed_shape) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t M = 1024;
    constexpr size_t N = P * 32;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto pack_input =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<M, N / P>);
    ntt::pack(ntt_input, pack_input, ntt::fixed_shape_v<1>);
    auto pack_output =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<16, 16>);
    ntt::slice(pack_input, pack_output, fixed_shape_v<0, 0>,
               fixed_shape_v<16, 16>, ntt::fixed_shape_v<0, 1>,
               ntt::fixed_shape_v<1, 1>);
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<16, 16 * P>);
    ntt::unpack(pack_output, ntt_output1, ntt::fixed_shape_v<1>);

    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<16, 16 * P>);
    // ort
    int64_t starts_buf[] = {0, 0};
    int64_t stops_buf[] = {16, 16 * P};
    int64_t axes_buf[] = {0, 1};
    int64_t steps_buf[] = {1, 1};
    RUN_ORT_SLICE2(starts_buf, stops_buf, axes_buf, steps_buf, 2)
}

TEST(SliceTestFloat, Pack_ranked_shape) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t M = 1024;
    constexpr size_t N = P * 32;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    auto ntt_input = ntt::make_tensor<float>(ntt::make_shape(M, N));
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto pack_input =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(M, N / P));
    ntt::pack(ntt_input, pack_input, ntt::fixed_shape_v<1>);
    auto pack_output =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(16, 16));
    ntt::slice(pack_input, pack_output, fixed_shape_v<0, 0>,
               fixed_shape_v<16, 16>, ntt::fixed_shape_v<0, 1>,
               ntt::fixed_shape_v<1, 1>);
    auto ntt_output1 = ntt::make_tensor<float>(ntt::make_shape(16, 16 * P));
    ntt::unpack(pack_output, ntt_output1, ntt::fixed_shape_v<1>);

    auto ntt_output2 = ntt::make_tensor<float>(ntt::make_shape(16, 16 * P));
    // ort
    int64_t starts_buf[] = {0, 0};
    int64_t stops_buf[] = {16, 16 * P};
    int64_t axes_buf[] = {0, 1};
    int64_t steps_buf[] = {1, 1};
    RUN_ORT_SLICE2(starts_buf, stops_buf, axes_buf, steps_buf, 2)
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}