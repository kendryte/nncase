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
#include <nncase/float8.h>
#include <nncase/ntt/ntt.h>
#include <ortki/operators.h>

using namespace nncase;
using namespace ortki;

TEST(WhereTestFloat, fixed_fixed_fixed_unpack) {
    constexpr size_t n = 1;
    constexpr size_t c = 1;
    constexpr size_t h = 2;
    constexpr size_t w = 2;
    // constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    float min_input = -100.0f;
    float max_input = 100.0f;

    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<n, c, h, w>>;
    using tensor_type2 = ntt::tensor<bool, ntt::fixed_shape<n, c, h, w>>;

    std::unique_ptr<tensor_type2> condition(new tensor_type2);
    NttTest::init_tensor(*condition, 0, 1);

    std::unique_ptr<tensor_type1> ntt_input1(new tensor_type1);
    NttTest::init_tensor(*ntt_input1, min_input, max_input);

    std::unique_ptr<tensor_type1> ntt_input2(new tensor_type1);
    NttTest::init_tensor(*ntt_input2, min_input, max_input);

    std::unique_ptr<tensor_type1> ntt_output1(new tensor_type1);
    // ntt
    ntt::where(*condition, *ntt_input1, *ntt_input2, *ntt_output1);

    // ort
    auto ort_condition = NttTest::ntt2ort(*condition);
    auto ort_input1 = NttTest::ntt2ort(*ntt_input1);
    auto ort_input2 = NttTest::ntt2ort(*ntt_input2);
    auto ort_output = ortki_Where(ort_condition, ort_input1, ort_input2);

    // compare
    std::unique_ptr<tensor_type1> ntt_output2(new tensor_type1);
    NttTest::ort2ntt(ort_output, *ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));
}

TEST(WhereTestFloat, scalar_fixed_fixed_unpack) {
    constexpr size_t n = 32;
    constexpr size_t c = 32;
    constexpr size_t h = 32;
    constexpr size_t w = 32;
    // constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    float min_input = -100.0f;
    float max_input = 100.0f;

    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<n, c, h, w>>;
    using tensor_type2 = ntt::tensor<bool, ntt::fixed_shape<1>>;

    std::unique_ptr<tensor_type2> condition(new tensor_type2);
    NttTest::init_tensor(*condition, 0, 1);

    std::unique_ptr<tensor_type1> ntt_input1(new tensor_type1);
    NttTest::init_tensor(*ntt_input1, min_input, max_input);

    std::unique_ptr<tensor_type1> ntt_input2(new tensor_type1);
    NttTest::init_tensor(*ntt_input2, min_input, max_input);

    std::unique_ptr<tensor_type1> ntt_output1(new tensor_type1);
    // ntt
    ntt::where(*condition, *ntt_input1, *ntt_input2, *ntt_output1);

    // ort
    auto ort_condition = NttTest::ntt2ort(*condition);
    auto ort_input1 = NttTest::ntt2ort(*ntt_input1);
    auto ort_input2 = NttTest::ntt2ort(*ntt_input2);
    auto ort_output = ortki_Where(ort_condition, ort_input1, ort_input2);

    // compare
    std::unique_ptr<tensor_type1> ntt_output2(new tensor_type1);
    NttTest::ort2ntt(ort_output, *ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));
}

TEST(WhereTestFloat, fixed_scalar_fixed_unpack) {
    constexpr size_t n = 32;
    constexpr size_t c = 32;
    constexpr size_t h = 32;
    constexpr size_t w = 32;
    // constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    float min_input = -100.0f;
    float max_input = 100.0f;

    using tensor_type0 = ntt::tensor<float, ntt::fixed_shape<1>>;
    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<n, c, h, w>>;
    using tensor_type2 = ntt::tensor<bool, ntt::fixed_shape<n, c, h, w>>;

    std::unique_ptr<tensor_type2> condition(new tensor_type2);
    NttTest::init_tensor(*condition, 0, 1);

    std::unique_ptr<tensor_type0> ntt_input1(new tensor_type0);
    NttTest::init_tensor(*ntt_input1, min_input, max_input);

    std::unique_ptr<tensor_type1> ntt_input2(new tensor_type1);
    NttTest::init_tensor(*ntt_input2, min_input, max_input);

    std::unique_ptr<tensor_type1> ntt_output1(new tensor_type1);
    // ntt
    ntt::where(*condition, *ntt_input1, *ntt_input2, *ntt_output1);

    // ort
    auto ort_condition = NttTest::ntt2ort(*condition);
    auto ort_input1 = NttTest::ntt2ort(*ntt_input1);
    auto ort_input2 = NttTest::ntt2ort(*ntt_input2);
    auto ort_output = ortki_Where(ort_condition, ort_input1, ort_input2);

    // compare
    std::unique_ptr<tensor_type1> ntt_output2(new tensor_type1);
    NttTest::ort2ntt(ort_output, *ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));
}

TEST(WhereTestFloat, fixed_fixed_scalar_unpack) {
    constexpr size_t n = 32;
    constexpr size_t c = 32;
    constexpr size_t h = 32;
    constexpr size_t w = 32;
    // constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    float min_input = -100.0f;
    float max_input = 100.0f;

    using tensor_type0 = ntt::tensor<float, ntt::fixed_shape<1>>;
    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<n, c, h, w>>;
    using tensor_type2 = ntt::tensor<bool, ntt::fixed_shape<n, c, h, w>>;

    std::unique_ptr<tensor_type2> condition(new tensor_type2);
    NttTest::init_tensor(*condition, 0, 1);

    std::unique_ptr<tensor_type1> ntt_input1(new tensor_type1);
    NttTest::init_tensor(*ntt_input1, min_input, max_input);

    std::unique_ptr<tensor_type0> ntt_input2(new tensor_type0);
    NttTest::init_tensor(*ntt_input2, min_input, max_input);

    std::unique_ptr<tensor_type1> ntt_output1(new tensor_type1);
    // ntt
    ntt::where(*condition, *ntt_input1, *ntt_input2, *ntt_output1);

    // ort
    auto ort_condition = NttTest::ntt2ort(*condition);
    auto ort_input1 = NttTest::ntt2ort(*ntt_input1);
    auto ort_input2 = NttTest::ntt2ort(*ntt_input2);
    auto ort_output = ortki_Where(ort_condition, ort_input1, ort_input2);

    // compare
    std::unique_ptr<tensor_type1> ntt_output2(new tensor_type1);
    NttTest::ort2ntt(ort_output, *ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));
}

TEST(WhereTestFloat, fixed_fixed_fixed_pack) {
    constexpr size_t n = 1;
    constexpr size_t c = 1;
    constexpr size_t h = 4;
    constexpr size_t w = 4;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    float min_input = -100.0f;
    float max_input = 100.0f;

    using tensor_type1_unpacked =
        ntt::tensor<float, ntt::fixed_shape<n, c, h, w>>;
    using tensor_type2_unpacked =
        ntt::tensor<bool, ntt::fixed_shape<n, c, h, w>>;

    using tensor_type1_packed =
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<n, c, h, w / P>>;
    using tensor_type2_packed =
        ntt::tensor<ntt::vector<bool, P>, ntt::fixed_shape<n, c, h, w / P>>;

    alignas(32) tensor_type2_unpacked condition;
    NttTest::init_tensor(condition, 0, 1);

    alignas(32) tensor_type1_unpacked ntt_input1, ntt_output1, ntt_output2;
    NttTest::init_tensor(ntt_input1, min_input, max_input);

    alignas(32) tensor_type1_unpacked ntt_input2;
    NttTest::init_tensor(ntt_input2, min_input, max_input);

    alignas(32) tensor_type1_packed ntt_input1_packed, ntt_input2_packed,
        ntt_output1_packed;
    alignas(32) tensor_type2_packed condition_packed;

    // ntt
    ntt::pack<3>(ntt_input1, ntt_input1_packed);
    ntt::pack<3>(ntt_input2, ntt_input2_packed);
    ntt::pack<3>(condition, condition_packed);

    ntt::where(condition_packed, ntt_input1_packed, ntt_input2_packed,
               ntt_output1_packed);
    ntt::unpack<3>(ntt_output1_packed, ntt_output1);

    // ort
    auto ort_condition = NttTest::ntt2ort(condition);
    auto ort_input1 = NttTest::ntt2ort(ntt_input1);
    auto ort_input2 = NttTest::ntt2ort(ntt_input2);
    auto ort_output = ortki_Where(ort_condition, ort_input1, ort_input2);

    // compare
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

// TEST(WhereTestFloat, fixed_fixed_scalar_pack) {
//     constexpr size_t n = 1;
//     constexpr size_t c = 1;
//     constexpr size_t h = 4;
//     constexpr size_t w = 4;
//     constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
//     float min_input = -100.0f;
//     float max_input = 100.0f;

//     using tensor_type1_unpacked =
//         ntt::tensor<float, ntt::fixed_shape<n, c, h, w>>;
//     using tensor_type2_unpacked =
//         ntt::tensor<bool, ntt::fixed_shape<n, c, h, w>>;

//     using tensor_type1_packed =
//         ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<n, c, h, w / P>>;
//     using tensor_type2_packed =
//         ntt::tensor<ntt::vector<bool, P>, ntt::fixed_shape<n, c, h, w / P>>;

//     alignas(32) tensor_type2_unpacked condition;
//     NttTest::init_tensor(condition, 0, 1);

//     alignas(32) tensor_type1_unpacked ntt_input1, ntt_output1, ntt_output2;
//     NttTest::init_tensor(ntt_input1, min_input, max_input);

//     alignas(32) ntt::tensor<float, ntt::fixed_shape<1>> ntt_input2;
//     NttTest::init_tensor(ntt_input2, min_input, max_input);

//     alignas(32) tensor_type1_packed ntt_input1_packed, ntt_output1_packed;
//     alignas(32) tensor_type2_packed condition_packed;

//     // ntt
//     ntt::pack<3>(ntt_input1, ntt_input1_packed);
//     ntt::pack<3>(condition, condition_packed);

//     ntt::where(condition_packed, ntt_input1_packed, ntt_input2,
//                ntt_output1_packed);
//     ntt::unpack<3>(ntt_output1_packed, ntt_output1);

//     // ort
//     auto ort_condition = NttTest::ntt2ort(condition);
//     auto ort_input1 = NttTest::ntt2ort(ntt_input1);
//     auto ort_input2 = NttTest::ntt2ort(ntt_input2);
//     auto ort_output = ortki_Where(ort_condition, ort_input1, ort_input2);

//     // compare
//     NttTest::ort2ntt(ort_output, ntt_output2);

//     std::cout << "ntt_input1: " << std::endl;
//     for (int i = 0; i < 16; ++i) {
//         std::cout << ntt_input1.elements().data()[i] << " ";
//         if (i == 15) std::cout << "\n";
//     }
//     std::cout << "ntt_input2: " << std::endl;
//     for (int i = 0; i < 16; ++i) {
//         std::cout << ntt_input2.elements().data()[i] << " ";
//         if (i == 15) std::cout << "\n";
//     }
//     std::cout << "condition: " << std::endl;
//     for (int i = 0; i < 16; ++i) {
//         std::cout << condition.elements().data()[i] << " ";
//         if (i == 15) std::cout << "\n";
//     }
//     std::cout << "ntt_output1: " << std::endl;
//     for (int i = 0; i < 16; ++i) {
//         std::cout << ntt_output1.elements().data()[i] << " ";
//         if (i == 15) std::cout << "\n";
//     }
//     std::cout << "ntt_output2: " << std::endl;
//     for (int i = 0; i < 16; ++i) {
//         std::cout << ntt_output2.elements().data()[i] << " ";
//         if (i == 15) std::cout << "\n";
//     }


//     EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
// }

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
