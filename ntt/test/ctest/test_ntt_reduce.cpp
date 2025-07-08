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
#include "nncase/ntt/kernels/reduce.h"
#include "ntt_test.h"
#include "ortki_helper.h"
#include <gtest/gtest.h>
#include <iostream>
#include <nncase/ntt/ntt.h>
#include <ortki/operators.h>

using namespace nncase;
using namespace ortki;

#define NTT_REDUCE_VERIFY_REDUCEM_NOPACK(M, N, ntt_reduce_mode)                \
    /* init */                                                                 \
    auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);        \
    std::iota(ntt_input.elements().begin(), ntt_input.elements().end(), 0.f);  \
                                                                               \
    /* ntt */                                                                  \
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<1, N>);      \
    ntt::reduce_##ntt_reduce_mode(ntt_input, ntt_output1,                      \
                                  ntt::fixed_shape_v<0>);                      \
                                                                               \
    auto ntt_output1_view = ntt::make_tensor_view(ntt_output1.elements(),      \
                                                  ntt::fixed_shape_v<1, N>);   \
                                                                               \
    auto ntt_output2 = ntt::make_tensor_view(                                  \
        std::span<float, N>(golden_array, N), ntt::fixed_shape_v<1, N>);       \
                                                                               \
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_view, ntt_output2));

TEST(ReduceSumTestFloat, ReduceM_NoPack) {
    constexpr size_t M = 16;
    constexpr size_t N = 16;
    float golden_array[] = {1920, 1936, 1952, 1968, 1984, 2000, 2016, 2032,
                            2048, 2064, 2080, 2096, 2112, 2128, 2144, 2160};

    NTT_REDUCE_VERIFY_REDUCEM_NOPACK(M, N, sum)
}

TEST(ReduceMaxTestFloat, ReduceM_NoPack) {
    constexpr size_t M = 16;
    constexpr size_t N = 16;
    float golden_array[] = {240, 241, 242, 243, 244, 245, 246, 247,
                            248, 249, 250, 251, 252, 253, 254, 255};

    NTT_REDUCE_VERIFY_REDUCEM_NOPACK(M, N, max)
}

TEST(ReduceMinTestFloat, ReduceM_NoPack) {
    constexpr size_t M = 16;
    constexpr size_t N = 16;
    float golden_array[] = {0, 1, 2,  3,  4,  5,  6,  7,
                            8, 9, 10, 11, 12, 13, 14, 15};

    NTT_REDUCE_VERIFY_REDUCEM_NOPACK(M, N, min)
}

TEST(ReduceMeanTestFloat, ReduceM_NoPack) {
    constexpr size_t M = 16;
    constexpr size_t N = 16;
    float golden_array[] = {120, 121, 122, 123, 124, 125, 126, 127,
                            128, 129, 130, 131, 132, 133, 134, 135};

    NTT_REDUCE_VERIFY_REDUCEM_NOPACK(M, N, mean)
}

#define NTT_PACKED_REDUCE_VERIFY_REDUCEM_PACKM(M, N, ntt_reduce_mode)          \
    /* init */                                                                 \
    alignas(32) auto ntt_input =                                               \
        ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);                     \
    std::iota(ntt_input.elements().begin(), ntt_input.elements().end(), 0.f);  \
                                                                               \
    alignas(32) auto ntt_input_pack =                                          \
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<M / P, N>); \
    ntt::pack(ntt_input, ntt_input_pack, ntt::fixed_shape_v<0>);               \
                                                                               \
    /* ntt */                                                                  \
    alignas(32) auto ntt_output1 =                                             \
        ntt::make_tensor<float>(ntt::fixed_shape_v<1, N>);                     \
    ntt::reduce_##ntt_reduce_mode(ntt_input_pack, ntt_output1,                 \
                                  ntt::fixed_shape_v<0>,                       \
                                  ntt::fixed_shape_v<0>);                      \
                                                                               \
    alignas(32) auto ntt_output1_view = ntt::make_tensor_view(                 \
        ntt_output1.elements(), ntt::fixed_shape_v<1, N>);                     \
                                                                               \
    alignas(32) auto ntt_output2 = ntt::make_tensor_view(                      \
        std::span<float, N>(golden_array, N), ntt::fixed_shape_v<1, N>);       \
                                                                               \
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_view, ntt_output2));

TEST(ReduceSumTestFloat, ReduceM_PackM) {
    constexpr size_t M = 16;
    constexpr size_t N = 16;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    alignas(32) float golden_array[] = {1920, 1936, 1952, 1968, 1984, 2000,
                                        2016, 2032, 2048, 2064, 2080, 2096,
                                        2112, 2128, 2144, 2160};

    NTT_PACKED_REDUCE_VERIFY_REDUCEM_PACKM(M, N, sum)
}

TEST(ReduceMaxTestFloat, ReduceM_PackM) {
    constexpr size_t M = 16;
    constexpr size_t N = 16;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    alignas(32) float golden_array[] = {240, 241, 242, 243, 244, 245, 246, 247,
                                        248, 249, 250, 251, 252, 253, 254, 255};

    NTT_PACKED_REDUCE_VERIFY_REDUCEM_PACKM(M, N, max)
}

TEST(ReduceMinTestFloat, ReduceM_PackM) {
    constexpr size_t M = 16;
    constexpr size_t N = 16;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    float golden_array[] = {0, 1, 2,  3,  4,  5,  6,  7,
                            8, 9, 10, 11, 12, 13, 14, 15};

    NTT_PACKED_REDUCE_VERIFY_REDUCEM_PACKM(M, N, min)
}

TEST(ReduceMeanTestFloat, ReduceM_PackM) {
    constexpr size_t M = 16;
    constexpr size_t N = 16;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    float golden_array[] = {120, 121, 122, 123, 124, 125, 126, 127,
                            128, 129, 130, 131, 132, 133, 134, 135};

    NTT_PACKED_REDUCE_VERIFY_REDUCEM_PACKM(M, N, mean)
}

#define NTT_REDUCE_VERIFY_REDUCEN_NOPACK(M, N, ntt_reduce_mode)                \
    /* init */                                                                 \
    auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);        \
    std::iota(ntt_input.elements().begin(), ntt_input.elements().end(), 0.f);  \
                                                                               \
    /* ntt */                                                                  \
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, 1>);      \
    ntt::reduce_##ntt_reduce_mode(ntt_input, ntt_output1,                      \
                                  ntt::fixed_shape_v<1>);                      \
                                                                               \
    auto ntt_output1_view = ntt::make_tensor_view(ntt_output1.elements(),      \
                                                  ntt::fixed_shape_v<M, 1>);   \
                                                                               \
    auto ntt_output2 = ntt::make_tensor_view(                                  \
        std::span<float, M>(golden_array, N), ntt::fixed_shape_v<M, 1>);       \
                                                                               \
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_view, ntt_output2));

TEST(ReduceSumTestFloat, ReduceN_NoPack) {
    constexpr size_t M = 16;
    constexpr size_t N = 16;

    float golden_array[] = {120,  376,  632,  888,  1144, 1400, 1656, 1912,
                            2168, 2424, 2680, 2936, 3192, 3448, 3704, 3960};
    NTT_REDUCE_VERIFY_REDUCEN_NOPACK(M, N, sum)
}

TEST(ReduceMaxTestFloat, ReduceN_NoPack) {
    constexpr size_t M = 16;
    constexpr size_t N = 16;

    float golden_array[] = {15,  31,  47,  63,  79,  95,  111, 127,
                            143, 159, 175, 191, 207, 223, 239, 255};
    NTT_REDUCE_VERIFY_REDUCEN_NOPACK(M, N, max)
}

TEST(ReduceMinTestFloat, ReduceN_NoPack) {
    constexpr size_t M = 16;
    constexpr size_t N = 16;

    float golden_array[] = {0,   16,  32,  48,  64,  80,  96,  112,
                            128, 144, 160, 176, 192, 208, 224, 240};
    NTT_REDUCE_VERIFY_REDUCEN_NOPACK(M, N, min)
}

TEST(ReduceMeanTestFloat, ReduceN_NoPack) {
    constexpr size_t M = 16;
    constexpr size_t N = 16;

    float golden_array[] = {7.5,   23.5,  39.5,  55.5,  71.5,  87.5,
                            103.5, 119.5, 135.5, 151.5, 167.5, 183.5,
                            199.5, 215.5, 231.5, 247.5};
    NTT_REDUCE_VERIFY_REDUCEN_NOPACK(M, N, mean)
}

#define NTT_PACKED_REDUCE_VERIFY_REDUCEN_PACKN(M, N, ntt_reduce_mode)          \
    /* init */                                                                 \
    alignas(32) auto ntt_input =                                               \
        ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);                     \
    std::iota(ntt_input.elements().begin(), ntt_input.elements().end(), 0.f);  \
                                                                               \
    alignas(32) auto ntt_input_pack =                                          \
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<M, N / P>); \
    ntt::pack(ntt_input, ntt_input_pack, ntt::fixed_shape_v<1>);               \
                                                                               \
    /* ntt */                                                                  \
    alignas(32) auto ntt_output1 =                                             \
        ntt::make_tensor<float>(ntt::fixed_shape_v<M, 1>);                     \
    ntt::reduce_##ntt_reduce_mode(ntt_input_pack, ntt_output1,                 \
                                  ntt::fixed_shape_v<1>,                       \
                                  ntt::fixed_shape_v<1>);                      \
                                                                               \
    alignas(32) auto ntt_output1_view = ntt::make_tensor_view(                 \
        ntt_output1.elements(), ntt::fixed_shape_v<M, 1>);                     \
                                                                               \
    alignas(32) auto ntt_output2 = ntt::make_tensor_view(                      \
        std::span<float, N>(golden_array, N), ntt::fixed_shape_v<M, 1>);       \
                                                                               \
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_view, ntt_output2));

TEST(ReduceSumTestFloat, ReduceN_PackN) {
    constexpr size_t M = 16;
    constexpr size_t N = 16;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    float golden_array[] = {120,  376,  632,  888,  1144, 1400, 1656, 1912,
                            2168, 2424, 2680, 2936, 3192, 3448, 3704, 3960};
    NTT_PACKED_REDUCE_VERIFY_REDUCEN_PACKN(M, N, sum)
}

TEST(ReduceMaxTestFloat, ReduceN_PackN) {
    constexpr size_t M = 16;
    constexpr size_t N = 16;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    float golden_array[] = {15,  31,  47,  63,  79,  95,  111, 127,
                            143, 159, 175, 191, 207, 223, 239, 255};
    NTT_PACKED_REDUCE_VERIFY_REDUCEN_PACKN(M, N, max)
}

TEST(ReduceMinTestFloat, ReduceN_PackN) {
    constexpr size_t M = 16;
    constexpr size_t N = 16;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    float golden_array[] = {0,   16,  32,  48,  64,  80,  96,  112,
                            128, 144, 160, 176, 192, 208, 224, 240};
    NTT_PACKED_REDUCE_VERIFY_REDUCEN_PACKN(M, N, min)
}

TEST(ReduceMeanTestFloat, ReduceN_PackN) {
    constexpr size_t M = 16;
    constexpr size_t N = 16;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    float golden_array[] = {7.5,   23.5,  39.5,  55.5,  71.5,  87.5,
                            103.5, 119.5, 135.5, 151.5, 167.5, 183.5,
                            199.5, 215.5, 231.5, 247.5};
    NTT_PACKED_REDUCE_VERIFY_REDUCEN_PACKN(M, N, mean)
}

#define NTT_PACKED_REDUCE_VERIFY_REDUCEMN_NOPACK(M, N, ntt_reduce_mode)        \
    /* init */                                                                 \
    alignas(32) auto ntt_input =                                               \
        ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);                     \
    std::iota(ntt_input.elements().begin(), ntt_input.elements().end(), 0.f);  \
                                                                               \
    /* ntt */                                                                  \
    alignas(32) auto ntt_output1 =                                             \
        ntt::make_tensor<float>(ntt::fixed_shape_v<1, 1>);                     \
    ntt::reduce_##ntt_reduce_mode(ntt_input, ntt_output1,                      \
                                  ntt::fixed_shape_v<0, 1>);                   \
                                                                               \
    alignas(32) auto ntt_output1_view = ntt::make_tensor_view(                 \
        ntt_output1.elements(), ntt::fixed_shape_v<1, 1>);                     \
                                                                               \
    alignas(32) auto ntt_output2 = ntt::make_tensor_view(                      \
        std::span<float, 1>(golden_array, 1), ntt::fixed_shape_v<1, 1>);       \
                                                                               \
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_view, ntt_output2));

TEST(ReduceSumTestFloat, ReduceMN_NoPack) {
    constexpr size_t M = 16;
    constexpr size_t N = 16;
    float golden_array[] = {32640};

    NTT_PACKED_REDUCE_VERIFY_REDUCEMN_NOPACK(M, N, sum)
}

TEST(ReduceMaxTestFloat, ReduceMN_NoPack) {
    constexpr size_t M = 16;
    constexpr size_t N = 16;

    float golden_array[] = {255};

    NTT_PACKED_REDUCE_VERIFY_REDUCEMN_NOPACK(M, N, max)
}

TEST(ReduceMinTestFloat, ReduceMN_NoPack) {
    constexpr size_t M = 16;
    constexpr size_t N = 16;

    float golden_array[] = {0};

    NTT_PACKED_REDUCE_VERIFY_REDUCEMN_NOPACK(M, N, min)
}

TEST(ReduceMeanTestFloat, ReduceMN_NoPack) {
    constexpr size_t M = 16;
    constexpr size_t N = 16;

    float golden_array[] = {127.5};

    NTT_PACKED_REDUCE_VERIFY_REDUCEMN_NOPACK(M, N, mean)
}

#define NTT_PACKED_REDUCE_VERIFY_REDUCEMN_PACKM(M, N, ntt_reduce_mode)         \
    /* init */                                                                 \
    alignas(32) auto ntt_input =                                               \
        ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);                     \
    std::iota(ntt_input.elements().begin(), ntt_input.elements().end(), 0.f);  \
                                                                               \
    alignas(32) auto ntt_input_pack =                                          \
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<M / P, N>); \
    ntt::pack(ntt_input, ntt_input_pack, ntt::fixed_shape_v<0>);               \
                                                                               \
    /* ntt */                                                                  \
    alignas(32) auto ntt_output1 =                                             \
        ntt::make_tensor<float>(ntt::fixed_shape_v<1, 1>);                     \
    ntt::reduce_##ntt_reduce_mode(ntt_input_pack, ntt_output1,                 \
                                  ntt::fixed_shape_v<0, 1>,                    \
                                  ntt::fixed_shape_v<0>);                      \
                                                                               \
    alignas(32) auto ntt_output1_view = ntt::make_tensor_view(                 \
        ntt_output1.elements(), ntt::fixed_shape_v<1, 1>);                     \
                                                                               \
    alignas(32) auto ntt_output2 = ntt::make_tensor_view(                      \
        std::span<float, 1>(golden_array, 1), ntt::fixed_shape_v<1, 1>);       \
                                                                               \
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_view, ntt_output2));

TEST(ReduceSumTestFloat, ReduceMN_PackM) {
    constexpr size_t M = 16;
    constexpr size_t N = 16;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    float golden_array[] = {32640};

    NTT_PACKED_REDUCE_VERIFY_REDUCEMN_PACKM(M, N, sum)
}

TEST(ReduceMaxTestFloat, ReduceMN_PackM) {
    constexpr size_t M = 16;
    constexpr size_t N = 16;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    float golden_array[] = {255};

    NTT_PACKED_REDUCE_VERIFY_REDUCEMN_PACKM(M, N, max)
}

TEST(ReduceMinTestFloat, ReduceMN_PackM) {
    constexpr size_t M = 16;
    constexpr size_t N = 16;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    float golden_array[] = {0};

    NTT_PACKED_REDUCE_VERIFY_REDUCEMN_PACKM(M, N, min)
}

TEST(ReduceMeanTestFloat, ReduceMN_PackM) {
    constexpr size_t M = 16;
    constexpr size_t N = 16;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    float golden_array[] = {127.5};

    NTT_PACKED_REDUCE_VERIFY_REDUCEMN_PACKM(M, N, mean)
}

#define NTT_PACKED_REDUCE_VERIFY_REDUCEMN_PACKN(M, N, ntt_reduce_mode)         \
    /* init */                                                                 \
    alignas(32) auto ntt_input =                                               \
        ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);                     \
    std::iota(ntt_input.elements().begin(), ntt_input.elements().end(), 0.f);  \
                                                                               \
    alignas(32) auto ntt_input_pack =                                          \
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<M, N / P>); \
    ntt::pack(ntt_input, ntt_input_pack, ntt::fixed_shape_v<1>);               \
                                                                               \
    /* ntt */                                                                  \
    alignas(32) auto ntt_output1 =                                             \
        ntt::make_tensor<float>(ntt::fixed_shape_v<1, 1>);                     \
    ntt::reduce_##ntt_reduce_mode(ntt_input_pack, ntt_output1,                 \
                                  ntt::fixed_shape_v<0, 1>,                    \
                                  ntt::fixed_shape_v<1>);                      \
                                                                               \
    alignas(32) auto ntt_output1_view = ntt::make_tensor_view(                 \
        ntt_output1.elements(), ntt::fixed_shape_v<1, 1>);                     \
                                                                               \
    alignas(32) auto ntt_output2 = ntt::make_tensor_view(                      \
        std::span<float, 1>(golden_array, 1), ntt::fixed_shape_v<1, 1>);       \
                                                                               \
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_view, ntt_output2));

TEST(ReduceSumTestFloat, ReduceMN_PackN) {
    constexpr size_t M = 16;
    constexpr size_t N = 16;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    float golden_array[] = {32640};

    NTT_PACKED_REDUCE_VERIFY_REDUCEMN_PACKN(M, N, sum)
}

TEST(ReduceMaxTestFloat, ReduceMN_PackN) {
    constexpr size_t M = 16;
    constexpr size_t N = 16;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    float golden_array[] = {255};

    NTT_PACKED_REDUCE_VERIFY_REDUCEMN_PACKN(M, N, max)
}

TEST(ReduceMinTestFloat, ReduceMN_PackN) {
    constexpr size_t M = 16;
    constexpr size_t N = 16;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    float golden_array[] = {0};

    NTT_PACKED_REDUCE_VERIFY_REDUCEMN_PACKN(M, N, min)
}

TEST(ReduceMeanTestFloat, ReduceMN_PackN) {
    constexpr size_t M = 16;
    constexpr size_t N = 16;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    float golden_array[] = {127.5};

    NTT_PACKED_REDUCE_VERIFY_REDUCEMN_PACKN(M, N, mean)
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
