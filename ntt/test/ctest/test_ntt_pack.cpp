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

TEST(PackTestFloat, fixed_shape_dim_N) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2 * P;
    constexpr size_t C = P;
    constexpr size_t H = P;
    constexpr size_t W = P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<N, C, H, W>>;
    using tensor_type2 =
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<N / P, C, H, W>>;
    alignas(32) tensor_type1 ntt_input;
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) tensor_type2 ntt_output1;
    ntt::pack<0>(ntt_input, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t data[] = {2, N / 2, C, H, W};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto tmp = ortki_Reshape(ort_input, shape, 0);
    int64_t perms[] = {0, 2, 3, 4, 1};
    auto ort_output = ortki_Transpose(tmp, perms, std::size(perms));

    // compare
    alignas(32) tensor_type2 ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_shape_dim_C) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P;
    constexpr size_t C = 2 * P;
    constexpr size_t H = P;
    constexpr size_t W = P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<N, C, H, W>>;
    using tensor_type2 =
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<N, C / P, H, W>>;
    alignas(32) tensor_type1 ntt_input;
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) tensor_type2 ntt_output1;
    ntt::pack<1>(ntt_input, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t data[] = {N, 2, C / 2, H, W};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto tmp = ortki_Reshape(ort_input, shape, 0);
    int64_t perms[] = {0, 1, 3, 4, 2};
    auto ort_output = ortki_Transpose(tmp, perms, std::size(perms));

    // compare
    alignas(32) tensor_type2 ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_shape_dim_H) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P;
    constexpr size_t C = P;
    constexpr size_t H = P * 2;
    constexpr size_t W = P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<N, C, H, W>>;
    using tensor_type2 =
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<N, C, H / P, W>>;
    alignas(32) tensor_type1 ntt_input;
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) tensor_type2 ntt_output1;
    ntt::pack<2>(ntt_input, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);

    int64_t data[] = {N, C, 2, H / 2, W};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto tmp = ortki_Reshape(ort_input, shape, 0);
    int64_t perms[] = {0, 1, 2, 4, 3};
    auto ort_output = ortki_Transpose(tmp, perms, std::size(perms));

    // compare
    alignas(32) tensor_type2 ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_shape_dim_W) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P;
    constexpr size_t C = P;
    constexpr size_t H = P;
    constexpr size_t W = 2 * P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<N, C, H, W>>;
    using tensor_type2 =
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<N, C, H, W / P>>;
    alignas(32) tensor_type1 ntt_input;
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) tensor_type2 ntt_output1;
    ntt::pack<3>(ntt_input, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t data[] = {N, C, H, 2, W / 2};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto ort_output = ortki_Reshape(ort_input, shape, 0);

    // compare
    alignas(32) tensor_type2 ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_shape_dim_N_C) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2 * P;
    constexpr size_t C = 2 * P;
    constexpr size_t H = P;
    constexpr size_t W = P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<N, C, H, W>>;
    using tensor_type2 = ntt::tensor<ntt::vector<float, P, P>,
                                     ntt::fixed_shape<N / P, C / P, H, W>>;
    alignas(32) tensor_type1 ntt_input;
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) tensor_type2 ntt_output1;
    ntt::pack<0, 1>(ntt_input, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t data[] = {2, N / 2, 2, C / 2, H, W};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto tmp = ortki_Reshape(ort_input, shape, 0);
    int64_t perms[] = {0, 2, 4, 5, 1, 3};
    auto ort_output = ortki_Transpose(tmp, perms, std::size(perms));

    // compare
    alignas(32) tensor_type2 ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_shape_dim_C_H) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P;
    constexpr size_t C = 2 * P;
    constexpr size_t H = 2 * P;
    constexpr size_t W = P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<N, C, H, W>>;
    using tensor_type2 = ntt::tensor<ntt::vector<float, P, P>,
                                     ntt::fixed_shape<N, C / P, H / P, W>>;
    alignas(32) tensor_type1 ntt_input;
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) tensor_type2 ntt_output1;
    ntt::pack<1, 2>(ntt_input, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);

    int64_t data[] = {N, 2, C / 2, 2, H / 2, W};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto tmp = ortki_Reshape(ort_input, shape, 0);
    int64_t perms[] = {0, 1, 3, 5, 2, 4};
    auto ort_output = ortki_Transpose(tmp, perms, std::size(perms));

    // compare
    alignas(32) tensor_type2 ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, fixed_shape_dim_H_W) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P;
    constexpr size_t C = P;
    constexpr size_t H = 2 * P;
    constexpr size_t W = 2 * P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<N, C, H, W>>;
    using tensor_type2 = ntt::tensor<ntt::vector<float, P, P>,
                                     ntt::fixed_shape<N, C, H / P, W / P>>;
    alignas(32) tensor_type1 ntt_input;
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) tensor_type2 ntt_output1;
    ntt::pack<2, 3>(ntt_input, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t data[] = {N, C, 2, H / 2, 2, W / 2};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto tmp = ortki_Reshape(ort_input, shape, 0);
    int64_t perms[] = {0, 1, 2, 4, 3, 5};
    auto ort_output = ortki_Transpose(tmp, perms, std::size(perms));

    // compare
    alignas(32) tensor_type2 ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, ranked_shape_dim_N) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2 * P;
    constexpr size_t C = P;
    constexpr size_t H = P;
    constexpr size_t W = P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    using tensor_type1 = ntt::tensor<float, ntt::ranked_shape<4>>;
    using tensor_type2 =
        ntt::tensor<ntt::vector<float, P>, ntt::ranked_shape<4>>;
    auto shape1 = ntt::make_ranked_shape(N, C, H, W);
    auto shape2 = ntt::make_ranked_shape(N / P, C, H, W);
    alignas(32) tensor_type1 ntt_input(shape1);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) tensor_type2 ntt_output1(shape2);
    ntt::pack<0>(ntt_input, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t data[] = {2, N / 2, C, H, W};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto tmp = ortki_Reshape(ort_input, shape, 0);
    int64_t perms[] = {0, 2, 3, 4, 1};
    auto ort_output = ortki_Transpose(tmp, perms, std::size(perms));

    // compare
    alignas(32) tensor_type2 ntt_output2(shape2);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, ranked_shape_dim_C) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P;
    constexpr size_t C = 2 * P;
    constexpr size_t H = P;
    constexpr size_t W = P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    using tensor_type1 = ntt::tensor<float, ntt::ranked_shape<4>>;
    using tensor_type2 =
        ntt::tensor<ntt::vector<float, P>, ntt::ranked_shape<4>>;
    auto shape1 = ntt::make_ranked_shape(N, C, H, W);
    auto shape2 = ntt::make_ranked_shape(N, C / P, H, W);
    alignas(32) tensor_type1 ntt_input(shape1);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) tensor_type2 ntt_output1(shape2);
    ntt::pack<1>(ntt_input, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t data[] = {N, 2, C / 2, H, W};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto tmp = ortki_Reshape(ort_input, shape, 0);
    int64_t perms[] = {0, 1, 3, 4, 2};
    auto ort_output = ortki_Transpose(tmp, perms, std::size(perms));

    // compare
    alignas(32) tensor_type2 ntt_output2(shape2);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, ranked_shape_dim_H) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P;
    constexpr size_t C = P;
    constexpr size_t H = 2 * P;
    constexpr size_t W = P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    using tensor_type1 = ntt::tensor<float, ntt::ranked_shape<4>>;
    using tensor_type2 =
        ntt::tensor<ntt::vector<float, P>, ntt::ranked_shape<4>>;
    auto shape1 = ntt::make_ranked_shape(N, C, H, W);
    auto shape2 = ntt::make_ranked_shape(N, C, H / P, W);
    alignas(32) tensor_type1 ntt_input(shape1);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) tensor_type2 ntt_output1(shape2);
    ntt::pack<2>(ntt_input, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t data[] = {N, C, 2, H / 2, W};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto tmp = ortki_Reshape(ort_input, shape, 0);
    int64_t perms[] = {0, 1, 2, 4, 3};
    auto ort_output = ortki_Transpose(tmp, perms, std::size(perms));

    // compare
    alignas(32) tensor_type2 ntt_output2(shape2);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, ranked_shape_dim_W) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P;
    constexpr size_t C = P;
    constexpr size_t H = P;
    constexpr size_t W = 2 * P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    using tensor_type1 = ntt::tensor<float, ntt::ranked_shape<4>>;
    using tensor_type2 =
        ntt::tensor<ntt::vector<float, P>, ntt::ranked_shape<4>>;
    auto shape1 = ntt::make_ranked_shape(N, C, H, W);
    auto shape2 = ntt::make_ranked_shape(N, C, H, W / P);
    alignas(32) tensor_type1 ntt_input(shape1);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) tensor_type2 ntt_output1(shape2);
    ntt::pack<3>(ntt_input, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t data[] = {N, C, H, 2, W / 2};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto ort_output = ortki_Reshape(ort_input, shape, 0);

    // compare
    alignas(32) tensor_type2 ntt_output2(shape2);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, ranked_shape_dim_N_C) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = 2 * P;
    constexpr size_t C = 2 * P;
    constexpr size_t H = P;
    constexpr size_t W = P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    using tensor_type1 = ntt::tensor<float, ntt::ranked_shape<4>>;
    using tensor_type2 =
        ntt::tensor<ntt::vector<float, P, P>, ntt::ranked_shape<4>>;
    auto shape1 = ntt::make_ranked_shape(N, C, H, W);
    auto shape2 = ntt::make_ranked_shape(N / P, C / P, H, W);
    alignas(32) tensor_type1 ntt_input(shape1);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) tensor_type2 ntt_output1(shape2);
    ntt::pack<0, 1>(ntt_input, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t data[] = {2, N / 2, 2, C / 2, H, W};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto tmp = ortki_Reshape(ort_input, shape, 0);
    int64_t perms[] = {0, 2, 4, 5, 1, 3};
    auto ort_output = ortki_Transpose(tmp, perms, std::size(perms));

    // compare
    alignas(32) tensor_type2 ntt_output2(shape2);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, ranked_shape_dim_C_H) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P;
    constexpr size_t C = 2 * P;
    constexpr size_t H = 2 * P;
    constexpr size_t W = P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    using tensor_type1 = ntt::tensor<float, ntt::ranked_shape<4>>;
    using tensor_type2 =
        ntt::tensor<ntt::vector<float, P, P>, ntt::ranked_shape<4>>;
    auto shape1 = ntt::make_ranked_shape(N, C, H, W);
    auto shape2 = ntt::make_ranked_shape(N, C / P, H / P, W);
    alignas(32) tensor_type1 ntt_input(shape1);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) tensor_type2 ntt_output1(shape2);
    ntt::pack<1, 2>(ntt_input, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t data[] = {N, 2, C / 2, 2, H / 2, W};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto tmp = ortki_Reshape(ort_input, shape, 0);
    int64_t perms[] = {0, 1, 3, 5, 2, 4};
    auto ort_output = ortki_Transpose(tmp, perms, std::size(perms));

    // compare
    alignas(32) tensor_type2 ntt_output2(shape2);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(PackTestFloat, ranked_shape_dim_H_W) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P;
    constexpr size_t C = P;
    constexpr size_t H = 2 * P;
    constexpr size_t W = 2 * P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    using tensor_type1 = ntt::tensor<float, ntt::ranked_shape<4>>;
    using tensor_type2 =
        ntt::tensor<ntt::vector<float, P, P>, ntt::ranked_shape<4>>;
    auto shape1 = ntt::make_ranked_shape(N, C, H, W);
    auto shape2 = ntt::make_ranked_shape(N, C, H / P, W / P);
    alignas(32) tensor_type1 ntt_input(shape1);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) tensor_type2 ntt_output1(shape2);
    ntt::pack<2, 3>(ntt_input, ntt_output1);

    // ort
    alignas(32) auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t data[] = {N, C, 2, H / 2, 2, W / 2};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto tmp = ortki_Reshape(ort_input, shape, 0);
    int64_t perms[] = {0, 1, 2, 4, 3, 5};
    auto ort_output = ortki_Transpose(tmp, perms, std::size(perms));

    // compare
    alignas(32) tensor_type2 ntt_output2(shape2);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}