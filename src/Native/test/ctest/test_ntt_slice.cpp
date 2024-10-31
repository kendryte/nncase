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

TEST(SliceTestFloat, NoPack_dim_1_step_eq_1) {
    float min_input = -10.0f;
    float max_input = 10.0f;

    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<32, 32>>;
    using tensor_type2 = ntt::tensor<float, ntt::fixed_shape<32, 4>>;

    // init
    std::unique_ptr<tensor_type1> ntt_input(new tensor_type1);
    NttTest::init_tensor(*ntt_input, min_input, max_input);

    // ntt
    std::unique_ptr<tensor_type2> ntt_output1(new tensor_type2);
    ntt::slice<ntt::fixed_shape<4>, ntt::fixed_shape<8>, ntt::fixed_shape<1>,
               ntt::fixed_shape<1>>(*ntt_input, *ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(*ntt_input);
    int64_t starts_buf[] = {4};
    int64_t stops_buf[] = {8};
    int64_t axes_buf[] = {1};
    int64_t steps_buf[] = {1};
    int64_t shape[] = {std::size(starts_buf)};
    auto starts = make_tensor(reinterpret_cast<void *>(starts_buf),
                              DataType_INT64, shape, 1);
    auto stops = make_tensor(reinterpret_cast<void *>(stops_buf),
                             DataType_INT64, shape, 1);
    auto axes = make_tensor(reinterpret_cast<void *>(axes_buf), DataType_INT64,
                            shape, 1);
    auto steps = make_tensor(reinterpret_cast<void *>(steps_buf),
                             DataType_INT64, shape, 1);
    auto ort_output = ortki_Slice(ort_input, starts, stops, axes, steps);

    // compare
    std::unique_ptr<tensor_type2> ntt_output2(new tensor_type2);
    NttTest::ort2ntt(ort_output, *ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));
}

TEST(SliceTestFloat, NoPack_dim_1_step_gt_1) {
    float min_input = -10.0f;
    float max_input = 10.0f;

    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<32, 32>>;
    using tensor_type2 = ntt::tensor<float, ntt::fixed_shape<32, 4>>;

    // init
    std::unique_ptr<tensor_type1> ntt_input(new tensor_type1);
    NttTest::init_tensor(*ntt_input, min_input, max_input);

    // ntt
    std::unique_ptr<tensor_type2> ntt_output1(new tensor_type2);
    ntt::slice<ntt::fixed_shape<4>, ntt::fixed_shape<12>, ntt::fixed_shape<1>,
               ntt::fixed_shape<2>>(*ntt_input, *ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(*ntt_input);
    int64_t starts_buf[] = {4};
    int64_t stops_buf[] = {12};
    int64_t axes_buf[] = {1};
    int64_t steps_buf[] = {2};
    int64_t shape[] = {std::size(starts_buf)};
    auto starts = make_tensor(reinterpret_cast<void *>(starts_buf),
                              DataType_INT64, shape, 1);
    auto stops = make_tensor(reinterpret_cast<void *>(stops_buf),
                             DataType_INT64, shape, 1);
    auto axes = make_tensor(reinterpret_cast<void *>(axes_buf), DataType_INT64,
                            shape, 1);
    auto steps = make_tensor(reinterpret_cast<void *>(steps_buf),
                             DataType_INT64, shape, 1);
    auto ort_output = ortki_Slice(ort_input, starts, stops, axes, steps);

    // compare
    std::unique_ptr<tensor_type2> ntt_output2(new tensor_type2);
    NttTest::ort2ntt(ort_output, *ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));
}

TEST(SliceTestFloat, NoPack_dim_0_step_eq_1) {
    float min_input = -10.0f;
    float max_input = 10.0f;

    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<32, 32>>;
    using tensor_type2 = ntt::tensor<float, ntt::fixed_shape<4, 32>>;

    // init
    std::unique_ptr<tensor_type1> ntt_input(new tensor_type1);
    NttTest::init_tensor(*ntt_input, min_input, max_input);

    // ntt
    std::unique_ptr<tensor_type2> ntt_output1(new tensor_type2);
    ntt::slice<ntt::fixed_shape<4>, ntt::fixed_shape<8>, ntt::fixed_shape<0>,
               ntt::fixed_shape<1>>(*ntt_input, *ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(*ntt_input);
    int64_t starts_buf[] = {4};
    int64_t stops_buf[] = {8};
    int64_t axes_buf[] = {0};
    int64_t steps_buf[] = {1};
    int64_t shape[] = {std::size(starts_buf)};
    auto starts = make_tensor(reinterpret_cast<void *>(starts_buf),
                              DataType_INT64, shape, 1);
    auto stops = make_tensor(reinterpret_cast<void *>(stops_buf),
                             DataType_INT64, shape, 1);
    auto axes = make_tensor(reinterpret_cast<void *>(axes_buf), DataType_INT64,
                            shape, 1);
    auto steps = make_tensor(reinterpret_cast<void *>(steps_buf),
                             DataType_INT64, shape, 1);
    auto ort_output = ortki_Slice(ort_input, starts, stops, axes, steps);

    // compare
    std::unique_ptr<tensor_type2> ntt_output2(new tensor_type2);
    NttTest::ort2ntt(ort_output, *ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));
}

TEST(SliceTestFloat, NoPack_dim_0_step_gt_1) {
    float min_input = -10.0f;
    float max_input = 10.0f;

    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<32, 32>>;
    using tensor_type2 = ntt::tensor<float, ntt::fixed_shape<4, 32>>;

    // init
    std::unique_ptr<tensor_type1> ntt_input(new tensor_type1);
    NttTest::init_tensor(*ntt_input, min_input, max_input);

    // ntt
    std::unique_ptr<tensor_type2> ntt_output1(new tensor_type2);
    ntt::slice<ntt::fixed_shape<4>, ntt::fixed_shape<12>, ntt::fixed_shape<0>,
               ntt::fixed_shape<2>>(*ntt_input, *ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(*ntt_input);
    int64_t starts_buf[] = {4};
    int64_t stops_buf[] = {12};
    int64_t axes_buf[] = {0};
    int64_t steps_buf[] = {2};
    int64_t shape[] = {std::size(starts_buf)};
    auto starts = make_tensor(reinterpret_cast<void *>(starts_buf),
                              DataType_INT64, shape, 1);
    auto stops = make_tensor(reinterpret_cast<void *>(stops_buf),
                             DataType_INT64, shape, 1);
    auto axes = make_tensor(reinterpret_cast<void *>(axes_buf), DataType_INT64,
                            shape, 1);
    auto steps = make_tensor(reinterpret_cast<void *>(steps_buf),
                             DataType_INT64, shape, 1);
    auto ort_output = ortki_Slice(ort_input, starts, stops, axes, steps);

    // compare
    std::unique_ptr<tensor_type2> ntt_output2(new tensor_type2);
    NttTest::ort2ntt(ort_output, *ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));
}

TEST(SliceTestFloat, NoPack_dim_0_1_step_eq_1) {
    float min_input = -10.0f;
    float max_input = 10.0f;

    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<32, 32>>;
    using tensor_type2 = ntt::tensor<float, ntt::fixed_shape<4, 4>>;

    // init
    std::unique_ptr<tensor_type1> ntt_input(new tensor_type1);
    NttTest::init_tensor(*ntt_input, min_input, max_input);

    // ntt
    std::unique_ptr<tensor_type2> ntt_output1(new tensor_type2);
    ntt::slice<ntt::fixed_shape<1, 4>, ntt::fixed_shape<5, 8>,
               ntt::fixed_shape<0, 1>, ntt::fixed_shape<1, 1>>(*ntt_input,
                                                               *ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(*ntt_input);
    int64_t starts_buf[] = {1, 4};
    int64_t stops_buf[] = {5, 8};
    int64_t axes_buf[] = {0, 1};
    int64_t steps_buf[] = {1, 1};
    int64_t shape[] = {std::size(starts_buf)};
    auto starts = make_tensor(reinterpret_cast<void *>(starts_buf),
                              DataType_INT64, shape, 1);
    auto stops = make_tensor(reinterpret_cast<void *>(stops_buf),
                             DataType_INT64, shape, 1);
    auto axes = make_tensor(reinterpret_cast<void *>(axes_buf), DataType_INT64,
                            shape, 1);
    auto steps = make_tensor(reinterpret_cast<void *>(steps_buf),
                             DataType_INT64, shape, 1);
    auto ort_output = ortki_Slice(ort_input, starts, stops, axes, steps);

    // compare
    std::unique_ptr<tensor_type2> ntt_output2(new tensor_type2);
    NttTest::ort2ntt(ort_output, *ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));
}

TEST(SliceTestFloat, NoPack_dim_0_1_step_gt_1) {
    float min_input = -10.0f;
    float max_input = 10.0f;

    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<32, 32>>;
    using tensor_type2 = ntt::tensor<float, ntt::fixed_shape<4, 4>>;

    // init
    std::unique_ptr<tensor_type1> ntt_input(new tensor_type1);
    NttTest::init_tensor(*ntt_input, min_input, max_input);

    // ntt
    std::unique_ptr<tensor_type2> ntt_output1(new tensor_type2);
    ntt::slice<ntt::fixed_shape<1, 4>, ntt::fixed_shape<8, 12>,
               ntt::fixed_shape<0, 1>, ntt::fixed_shape<2, 2>>(*ntt_input,
                                                               *ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(*ntt_input);
    int64_t starts_buf[] = {1, 4};
    int64_t stops_buf[] = {8, 12};
    int64_t axes_buf[] = {0, 1};
    int64_t steps_buf[] = {2, 2};
    int64_t shape[] = {std::size(starts_buf)};
    auto starts = make_tensor(reinterpret_cast<void *>(starts_buf),
                              DataType_INT64, shape, 1);
    auto stops = make_tensor(reinterpret_cast<void *>(stops_buf),
                             DataType_INT64, shape, 1);
    auto axes = make_tensor(reinterpret_cast<void *>(axes_buf), DataType_INT64,
                            shape, 1);
    auto steps = make_tensor(reinterpret_cast<void *>(steps_buf),
                             DataType_INT64, shape, 1);
    auto ort_output = ortki_Slice(ort_input, starts, stops, axes, steps);

    // compare
    std::unique_ptr<tensor_type2> ntt_output2(new tensor_type2);
    NttTest::ort2ntt(ort_output, *ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));
}

TEST(SliceTestFloat, NoPack_dim_multiple_step_eq_1) {
    float min_input = -10.0f;
    float max_input = 10.0f;

    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<1, 3, 32, 32>>;
    using tensor_type2 = ntt::tensor<float, ntt::fixed_shape<1, 2, 4, 4>>;

    // init
    std::unique_ptr<tensor_type1> ntt_input(new tensor_type1);
    NttTest::init_tensor(*ntt_input, min_input, max_input);

    // ntt
    std::unique_ptr<tensor_type2> ntt_output1(new tensor_type2);
    ntt::slice<ntt::fixed_shape<1, 1, 4>, ntt::fixed_shape<3, 5, 8>,
               ntt::fixed_shape<1, 2, 3>, ntt::fixed_shape<1, 1, 1>>(
        *ntt_input, *ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(*ntt_input);
    int64_t starts_buf[] = {1, 1, 4};
    int64_t stops_buf[] = {3, 5, 8};
    int64_t axes_buf[] = {1, 2, 3};
    int64_t steps_buf[] = {1, 1, 1};
    int64_t shape[] = {std::size(starts_buf)};
    auto starts = make_tensor(reinterpret_cast<void *>(starts_buf),
                              DataType_INT64, shape, 1);
    auto stops = make_tensor(reinterpret_cast<void *>(stops_buf),
                             DataType_INT64, shape, 1);
    auto axes = make_tensor(reinterpret_cast<void *>(axes_buf), DataType_INT64,
                            shape, 1);
    auto steps = make_tensor(reinterpret_cast<void *>(steps_buf),
                             DataType_INT64, shape, 1);
    auto ort_output = ortki_Slice(ort_input, starts, stops, axes, steps);

    // compare
    std::unique_ptr<tensor_type2> ntt_output2(new tensor_type2);
    NttTest::ort2ntt(ort_output, *ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));
}

TEST(SliceTestFloat, NoPack_dim_multiple_step_gt_1) {
    float min_input = -10.0f;
    float max_input = 10.0f;

    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<1, 3, 32, 32>>;
    using tensor_type2 = ntt::tensor<float, ntt::fixed_shape<1, 2, 5, 6>>;

    // init
    std::unique_ptr<tensor_type1> ntt_input(new tensor_type1);
    NttTest::init_tensor(*ntt_input, min_input, max_input);

    // ntt
    std::unique_ptr<tensor_type2> ntt_output1(new tensor_type2);
    ntt::slice<ntt::fixed_shape<0, 1, 4>, ntt::fixed_shape<3, 10, 16>,
               ntt::fixed_shape<1, 2, 3>, ntt::fixed_shape<2, 2, 2>>(
        *ntt_input, *ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(*ntt_input);
    int64_t starts_buf[] = {0, 1, 4};
    int64_t stops_buf[] = {3, 10, 16};
    int64_t axes_buf[] = {1, 2, 3};
    int64_t steps_buf[] = {2, 2, 2};
    int64_t shape[] = {std::size(starts_buf)};
    auto starts = make_tensor(reinterpret_cast<void *>(starts_buf),
                              DataType_INT64, shape, 1);
    auto stops = make_tensor(reinterpret_cast<void *>(stops_buf),
                             DataType_INT64, shape, 1);
    auto axes = make_tensor(reinterpret_cast<void *>(axes_buf), DataType_INT64,
                            shape, 1);
    auto steps = make_tensor(reinterpret_cast<void *>(steps_buf),
                             DataType_INT64, shape, 1);
    auto ort_output = ortki_Slice(ort_input, starts, stops, axes, steps);

    // compare
    std::unique_ptr<tensor_type2> ntt_output2(new tensor_type2);
    NttTest::ort2ntt(ort_output, *ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));
}

TEST(SliceTestFloat, Pack) {
    constexpr size_t M = 1024;
    constexpr size_t N = 1024;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    float min_input = -10.0f;
    float max_input = 10.0f;

    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    using tensor_type2 =
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M, N / P>>;
    using tensor_type3 =
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<16, 16>>;
    using tensor_type4 = ntt::tensor<float, ntt::fixed_shape<16, 64>>;

    // init
    std::unique_ptr<tensor_type1> ntt_input(new tensor_type1);
    NttTest::init_tensor(*ntt_input, min_input, max_input);

    // ntt
    std::unique_ptr<tensor_type2> pack_input(new tensor_type2);
    std::unique_ptr<tensor_type3> pack_output(new tensor_type3);
    ntt::pack<1>(*ntt_input, *pack_input);
    ntt::slice<ntt::fixed_shape<0, 0>, ntt::fixed_shape<16, 16>,
               ntt::fixed_shape<0, 1>, ntt::fixed_shape<1, 1>>(*pack_input,
                                                               *pack_output);
    std::unique_ptr<tensor_type4> ntt_output1(new tensor_type4);
    ntt::unpack<1>(*pack_output, *ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(*ntt_input);
    int64_t starts_buf[] = {0, 0};
    int64_t stops_buf[] = {16, 64};
    int64_t axes_buf[] = {0, 1};
    int64_t steps_buf[] = {1, 1};
    int64_t shape[] = {std::size(starts_buf)};
    auto starts = make_tensor(reinterpret_cast<void *>(starts_buf),
                              DataType_INT64, shape, 1);
    auto stops = make_tensor(reinterpret_cast<void *>(stops_buf),
                             DataType_INT64, shape, 1);
    auto axes = make_tensor(reinterpret_cast<void *>(axes_buf), DataType_INT64,
                            shape, 1);
    auto steps = make_tensor(reinterpret_cast<void *>(steps_buf),
                             DataType_INT64, shape, 1);
    auto ort_output = ortki_Slice(ort_input, starts, stops, axes, steps);

    // compare
    std::unique_ptr<tensor_type4> ntt_output2(new tensor_type4);
    NttTest::ort2ntt(ort_output, *ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}