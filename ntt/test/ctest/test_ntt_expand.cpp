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
#include <nncase/float8.h>
#include <nncase/ntt/ntt.h>
#include <ortki/operators.h>

using namespace nncase;
using namespace ortki;

TEST(Expand, W) {

    constexpr size_t N = 8;
    constexpr size_t C = 8;
    constexpr size_t H = 8;
    constexpr size_t W = 1;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<N, C, H, W>>;
    using tensor_type2 =
        ntt::tensor<float,
                    ntt::fixed_shape<expand_n, expand_c, expand_h, expand_w>>;

    alignas(32) tensor_type1 ntt_input;
    [[maybe_unused]] alignas(32) tensor_type2 ntt_output1;
    std::iota(ntt_input.elements().begin(), ntt_input.elements().end(), 0.f);

    ntt::expand(ntt_input, ntt_output1);

    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t data[] = {expand_n, expand_c, expand_h, expand_w};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto ort_output = ortki_Expand(ort_input, shape);

    alignas(32) tensor_type2 ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(Expand, H) {

    constexpr size_t N = 8;
    constexpr size_t C = 8;
    constexpr size_t H = 1;
    constexpr size_t W = 8;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<N, C, H, W>>;
    using tensor_type2 =
        ntt::tensor<float,
                    ntt::fixed_shape<expand_n, expand_c, expand_h, expand_w>>;

    alignas(32) tensor_type1 ntt_input;
    [[maybe_unused]] alignas(32) tensor_type2 ntt_output1;
    std::iota(ntt_input.elements().begin(), ntt_input.elements().end(), 0.f);

    ntt::expand(ntt_input, ntt_output1);

    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t data[] = {expand_n, expand_c, expand_h, expand_w};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto ort_output = ortki_Expand(ort_input, shape);

    alignas(32) tensor_type2 ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(Expand, C) {

    constexpr size_t N = 8;
    constexpr size_t C = 1;
    constexpr size_t H = 8;
    constexpr size_t W = 8;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<N, C, H, W>>;
    using tensor_type2 =
        ntt::tensor<float,
                    ntt::fixed_shape<expand_n, expand_c, expand_h, expand_w>>;

    alignas(32) tensor_type1 ntt_input;
    [[maybe_unused]] alignas(32) tensor_type2 ntt_output1;
    std::iota(ntt_input.elements().begin(), ntt_input.elements().end(), 0.f);

    ntt::expand(ntt_input, ntt_output1);

    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t data[] = {expand_n, expand_c, expand_h, expand_w};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto ort_output = ortki_Expand(ort_input, shape);

    alignas(32) tensor_type2 ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(Expand, N) {

    constexpr size_t N = 1;
    constexpr size_t C = 8;
    constexpr size_t H = 8;
    constexpr size_t W = 8;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<N, C, H, W>>;
    using tensor_type2 =
        ntt::tensor<float,
                    ntt::fixed_shape<expand_n, expand_c, expand_h, expand_w>>;

    alignas(32) tensor_type1 ntt_input;
    [[maybe_unused]] alignas(32) tensor_type2 ntt_output1;
    std::iota(ntt_input.elements().begin(), ntt_input.elements().end(), 0.f);

    ntt::expand(ntt_input, ntt_output1);

    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t data[] = {expand_n, expand_c, expand_h, expand_w};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto ort_output = ortki_Expand(ort_input, shape);

    alignas(32) tensor_type2 ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(Expand, NC) {

    constexpr size_t N = 1;
    constexpr size_t C = 1;
    constexpr size_t H = 8;
    constexpr size_t W = 8;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<N, C, H, W>>;
    using tensor_type2 =
        ntt::tensor<float,
                    ntt::fixed_shape<expand_n, expand_c, expand_h, expand_w>>;

    alignas(32) tensor_type1 ntt_input;
    [[maybe_unused]] alignas(32) tensor_type2 ntt_output1;
    std::iota(ntt_input.elements().begin(), ntt_input.elements().end(), 0.f);

    ntt::expand(ntt_input, ntt_output1);

    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t data[] = {expand_n, expand_c, expand_h, expand_w};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto ort_output = ortki_Expand(ort_input, shape);

    alignas(32) tensor_type2 ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(Expand, CH) {

    constexpr size_t N = 8;
    constexpr size_t C = 1;
    constexpr size_t H = 1;
    constexpr size_t W = 8;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<N, C, H, W>>;
    using tensor_type2 =
        ntt::tensor<float,
                    ntt::fixed_shape<expand_n, expand_c, expand_h, expand_w>>;

    alignas(32) tensor_type1 ntt_input;
    [[maybe_unused]] alignas(32) tensor_type2 ntt_output1;
    std::iota(ntt_input.elements().begin(), ntt_input.elements().end(), 0.f);

    ntt::expand(ntt_input, ntt_output1);

    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t data[] = {expand_n, expand_c, expand_h, expand_w};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto ort_output = ortki_Expand(ort_input, shape);

    alignas(32) tensor_type2 ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(Expand, HW) {

    constexpr size_t N = 8;
    constexpr size_t C = 8;
    constexpr size_t H = 1;
    constexpr size_t W = 1;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<N, C, H, W>>;
    using tensor_type2 =
        ntt::tensor<float,
                    ntt::fixed_shape<expand_n, expand_c, expand_h, expand_w>>;

    alignas(32) tensor_type1 ntt_input;
    [[maybe_unused]] alignas(32) tensor_type2 ntt_output1;
    std::iota(ntt_input.elements().begin(), ntt_input.elements().end(), 0.f);

    ntt::expand(ntt_input, ntt_output1);

    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t data[] = {expand_n, expand_c, expand_h, expand_w};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto ort_output = ortki_Expand(ort_input, shape);

    alignas(32) tensor_type2 ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(Expand, NH) {

    constexpr size_t N = 1;
    constexpr size_t C = 8;
    constexpr size_t H = 1;
    constexpr size_t W = 8;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<N, C, H, W>>;
    using tensor_type2 =
        ntt::tensor<float,
                    ntt::fixed_shape<expand_n, expand_c, expand_h, expand_w>>;

    alignas(32) tensor_type1 ntt_input;
    [[maybe_unused]] alignas(32) tensor_type2 ntt_output1;
    std::iota(ntt_input.elements().begin(), ntt_input.elements().end(), 0.f);

    ntt::expand(ntt_input, ntt_output1);

    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t data[] = {expand_n, expand_c, expand_h, expand_w};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto ort_output = ortki_Expand(ort_input, shape);

    alignas(32) tensor_type2 ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(Expand, CW) {

    constexpr size_t N = 8;
    constexpr size_t C = 1;
    constexpr size_t H = 8;
    constexpr size_t W = 1;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<N, C, H, W>>;
    using tensor_type2 =
        ntt::tensor<float,
                    ntt::fixed_shape<expand_n, expand_c, expand_h, expand_w>>;

    alignas(32) tensor_type1 ntt_input;
    [[maybe_unused]] alignas(32) tensor_type2 ntt_output1;
    std::iota(ntt_input.elements().begin(), ntt_input.elements().end(), 0.f);

    ntt::expand(ntt_input, ntt_output1);

    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t data[] = {expand_n, expand_c, expand_h, expand_w};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto ort_output = ortki_Expand(ort_input, shape);

    alignas(32) tensor_type2 ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(Expand, NW) {

    constexpr size_t N = 1;
    constexpr size_t C = 8;
    constexpr size_t H = 8;
    constexpr size_t W = 1;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<N, C, H, W>>;
    using tensor_type2 =
        ntt::tensor<float,
                    ntt::fixed_shape<expand_n, expand_c, expand_h, expand_w>>;

    alignas(32) tensor_type1 ntt_input;
    [[maybe_unused]] alignas(32) tensor_type2 ntt_output1;
    std::iota(ntt_input.elements().begin(), ntt_input.elements().end(), 0.f);

    ntt::expand(ntt_input, ntt_output1);

    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t data[] = {expand_n, expand_c, expand_h, expand_w};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto ort_output = ortki_Expand(ort_input, shape);

    alignas(32) tensor_type2 ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
