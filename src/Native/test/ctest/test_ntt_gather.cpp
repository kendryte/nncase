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

TEST(GatherTestFloat, no_pack_index_int32) {

    using tensor_a_type = ntt::tensor<float, ntt::fixed_shape<4, 5, 8, 3>>;
    using tensor_b_type = ntt::tensor<int32_t, ntt::fixed_shape<1, 3>>;
    using tensor_c_type = ntt::tensor<float, ntt::fixed_shape<4, 1, 3, 8, 3>>;

    int32_t index_array[] = {1, 0, 2};
    auto tb = ntt::tensor_view<int32_t, ntt::fixed_shape<1, 3>>(
        std::span<int32_t, 3>(index_array, 3));

    tensor_a_type ta;
    tensor_c_type tc;
    tensor_c_type td;
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
    ntt::gather<1>(ta, tb, tc);

    // ort
    auto ort_input = NttTest::ntt2ort(ta);
    auto ort_tb = NttTest::ntt2ort(tb);
    auto ort_output = ortki_Gather(ort_input, ort_tb, 1);

    // compare
    NttTest::ort2ntt(ort_output, td);
    EXPECT_TRUE(NttTest::compare_tensor(tc, td));
}

TEST(GatherTestFloat, no_pack_index_int64) {

    using tensor_a_type = ntt::tensor<float, ntt::fixed_shape<4, 5, 8, 3>>;
    using tensor_b_type = ntt::tensor<int64_t, ntt::fixed_shape<1, 3>>;
    using tensor_c_type = ntt::tensor<float, ntt::fixed_shape<4, 1, 3, 8, 3>>;

    int64_t index_array[] = {1, 0, 2};
    auto tb = ntt::tensor_view<int64_t, ntt::fixed_shape<1, 3>>(
        std::span<int64_t, 3>(index_array, 3));

    tensor_a_type ta;
    tensor_c_type tc;
    tensor_c_type td;
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
    ntt::gather<1>(ta, tb, tc);

    // ort
    auto ort_input = NttTest::ntt2ort(ta);
    auto ort_tb = NttTest::ntt2ort(tb);
    auto ort_output = ortki_Gather(ort_input, ort_tb, 1);

    // compare
    NttTest::ort2ntt(ort_output, td);
    EXPECT_TRUE(NttTest::compare_tensor(tc, td));
}

TEST(GatherTestFloat, pack1d_dim0_contiguous) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 32;
    constexpr size_t N = 128;
    constexpr size_t Period = 1;
    using tensor_a_type = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    using tensor_b_type = ntt::tensor<int64_t, ntt::fixed_shape<1, M / Period>>;
    using tensor_pa_type =
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M, N / P>>;
    using tensor_pc_type = ntt::tensor<ntt::vector<float, P>,
                                       ntt::fixed_shape<1, M / Period, N / P>>;

    tensor_a_type ta;
    tensor_b_type tb;
    tensor_pa_type pa;
    tensor_pc_type pc;
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
    std::iota(tb.elements().begin(), tb.elements().end(), 0);
    std::ranges::for_each(tb.elements(), [](int64_t &x) { x *= Period; });
    ntt::pack<1>(ta, pa);
    ntt::gather<0>(pa, tb, pc);

    // ort
    auto ort_input = NttTest::ntt2ort(pa);
    auto ort_tb = NttTest::ntt2ort(tb);
    auto ort_output = ortki_Gather(ort_input, ort_tb, 0);

    // compare
    tensor_pc_type pd;

    NttTest::ort2ntt(ort_output, pd);
    EXPECT_TRUE(NttTest::compare_tensor(pc, pd));
}

TEST(GatherTestFloat, pack1d_dim0_no_contiguous) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 32;
    constexpr size_t N = 128;
    constexpr size_t Period = 2;
    using tensor_a_type = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    using tensor_b_type = ntt::tensor<int64_t, ntt::fixed_shape<1, M / Period>>;
    using tensor_pa_type =
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M, N / P>>;
    using tensor_pc_type = ntt::tensor<ntt::vector<float, P>,
                                       ntt::fixed_shape<1, M / Period, N / P>>;

    tensor_a_type ta;
    tensor_b_type tb;
    tensor_pa_type pa;
    tensor_pc_type pc;
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
    std::iota(tb.elements().begin(), tb.elements().end(), 0);
    std::ranges::for_each(tb.elements(), [](int64_t &x) { x *= Period; });
    ntt::pack<1>(ta, pa);
    ntt::gather<0>(pa, tb, pc);

    // ort
    auto ort_input = NttTest::ntt2ort(pa);
    auto ort_tb = NttTest::ntt2ort(tb);
    auto ort_output = ortki_Gather(ort_input, ort_tb, 0);

    // compare
    tensor_pc_type pd;

    NttTest::ort2ntt(ort_output, pd);
    EXPECT_TRUE(NttTest::compare_tensor(pc, pd));
}

TEST(GatherTestFloat, pack1d_dim1_contiguous) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 4;
    constexpr size_t N = 512;
    constexpr size_t Period = 1;
    using tensor_a_type = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    using tensor_b_type =
        ntt::tensor<int64_t, ntt::fixed_shape<1, N / P / Period>>;
    using tensor_pa_type =
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M, N / P>>;
    using tensor_pc_type = ntt::tensor<ntt::vector<float, P>,
                                       ntt::fixed_shape<M, 1, N / P / Period>>;

    tensor_a_type ta;
    tensor_b_type tb;
    tensor_pa_type pa;
    tensor_pc_type pc;
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
    std::iota(tb.elements().begin(), tb.elements().end(), 0);
    std::ranges::for_each(tb.elements(), [](int64_t &x) { x *= Period; });
    ntt::pack<1>(ta, pa);
    ntt::gather<1>(pa, tb, pc);

    // ort
    auto ort_input = NttTest::ntt2ort(pa);
    auto ort_tb = NttTest::ntt2ort(tb);
    auto ort_output = ortki_Gather(ort_input, ort_tb, 1);

    // compare
    tensor_pc_type pd;

    NttTest::ort2ntt(ort_output, pd);
    EXPECT_TRUE(NttTest::compare_tensor(pc, pd));
}

TEST(GatherTestFloat, pack1d_dim1_no_contiguous) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 4;
    constexpr size_t N = 512;
    constexpr size_t Period = 2;
    using tensor_a_type = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    using tensor_b_type =
        ntt::tensor<int64_t, ntt::fixed_shape<1, N / P / Period>>;
    using tensor_pa_type =
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M, N / P>>;
    using tensor_pc_type = ntt::tensor<ntt::vector<float, P>,
                                       ntt::fixed_shape<M, 1, N / P / Period>>;

    tensor_a_type ta;
    tensor_b_type tb;
    tensor_pa_type pa;
    tensor_pc_type pc;
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
    std::iota(tb.elements().begin(), tb.elements().end(), 0);
    std::ranges::for_each(tb.elements(), [](int64_t &x) { x *= Period; });
    ntt::pack<1>(ta, pa);
    ntt::gather<1>(pa, tb, pc);

    // ort
    auto ort_input = NttTest::ntt2ort(pa);
    auto ort_tb = NttTest::ntt2ort(tb);
    auto ort_output = ortki_Gather(ort_input, ort_tb, 1);

    // compare
    tensor_pc_type pd;

    NttTest::ort2ntt(ort_output, pd);
    EXPECT_TRUE(NttTest::compare_tensor(pc, pd));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}