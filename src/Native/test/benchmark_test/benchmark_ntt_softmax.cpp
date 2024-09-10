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
#include <iomanip>
#include <nncase/ntt/ntt.h>

using namespace nncase;

void benchmark_ntt_softmax_no_pack_dim0() {
    constexpr size_t warmup_num = 10;
    constexpr size_t run_num = 3000;
    ntt::tensor<float, ntt::fixed_shape<32, 32>> buffer_1;
    ntt::tensor<float, ntt::fixed_shape<32, 32>> buffer_2;
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);

    for (size_t i = 0; i < warmup_num; i++) {
        packed_softmax<0>(buffer_1, buffer_2, ntt::fixed_shape<>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        packed_softmax<0>(buffer_1, buffer_2, ntt::fixed_shape<>{});
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

void benchmark_ntt_softmax_no_pack_dim1() {
    constexpr size_t warmup_num = 10;
    constexpr size_t run_num = 3000;
    ntt::tensor<float, ntt::fixed_shape<32, 32>> buffer_1;
    ntt::tensor<float, ntt::fixed_shape<32, 32>> buffer_2;
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);

    for (size_t i = 0; i < warmup_num; i++) {
        packed_softmax<1>(buffer_1, buffer_2, ntt::fixed_shape<>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        packed_softmax<1>(buffer_1, buffer_2, ntt::fixed_shape<>{});
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

void benchmark_ntt_softmax_pack0_dim0() {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t warmup_num = 10;
    constexpr size_t run_num = 3000;
    ntt::tensor<float, ntt::fixed_shape<32, 32>> buffer_1;
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<32 / P, 32>> buffer_1_p;
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<32 / P, 32>> buffer_2_p;
    ntt::tensor<float, ntt::fixed_shape<32, 32>> buffer_2_up;
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);
    pack<0>(buffer_1, buffer_1_p);

    for (size_t i = 0; i < warmup_num; i++) {
        packed_softmax<0>(buffer_1_p, buffer_2_p, ntt::fixed_shape<0>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        packed_softmax<0>(buffer_1_p, buffer_2_p, ntt::fixed_shape<0>{});
    }
    auto t2 = NttTest::get_cpu_cycle();
    unpack<0>(buffer_2_p, buffer_2_up);

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

void benchmark_ntt_softmax_pack0_dim1() {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t warmup_num = 10;
    constexpr size_t run_num = 3000;
    ntt::tensor<float, ntt::fixed_shape<32, 32>> buffer_1;
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<32 / P, 32>> buffer_1_p;
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<32 / P, 32>> buffer_2_p;
    ntt::tensor<float, ntt::fixed_shape<32, 32>> buffer_2_up;
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);
    pack<0>(buffer_1, buffer_1_p);

    for (size_t i = 0; i < warmup_num; i++) {
        packed_softmax<1>(buffer_1_p, buffer_2_p, ntt::fixed_shape<0>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        packed_softmax<1>(buffer_1_p, buffer_2_p, ntt::fixed_shape<0>{});
    }
    auto t2 = NttTest::get_cpu_cycle();
    unpack<0>(buffer_2_p, buffer_2_up);

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

void benchmark_ntt_softmax_pack1_dim0() {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t warmup_num = 10;
    constexpr size_t run_num = 3000;
    ntt::tensor<float, ntt::fixed_shape<32, 32>> buffer_1;
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<32, 32 / P>> buffer_1_p;
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<32, 32 / P>> buffer_2_p;
    ntt::tensor<float, ntt::fixed_shape<32, 32>> buffer_2_up;
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);
    pack<1>(buffer_1, buffer_1_p);

    for (size_t i = 0; i < warmup_num; i++) {
        packed_softmax<0>(buffer_1_p, buffer_2_p, ntt::fixed_shape<1>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        packed_softmax<0>(buffer_1_p, buffer_2_p, ntt::fixed_shape<1>{});
    }
    auto t2 = NttTest::get_cpu_cycle();
    unpack<1>(buffer_2_p, buffer_2_up);

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

void benchmark_ntt_softmax_pack1_dim1() {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t warmup_num = 10;
    constexpr size_t run_num = 3000;
    ntt::tensor<float, ntt::fixed_shape<32, 32>> buffer_1;
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<32, 32 / P>> buffer_1_p;
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<32, 32 / P>> buffer_2_p;
    ntt::tensor<float, ntt::fixed_shape<32, 32>> buffer_2_up;
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);
    pack<1>(buffer_1, buffer_1_p);

    for (size_t i = 0; i < warmup_num; i++) {
        packed_softmax<1>(buffer_1_p, buffer_2_p, ntt::fixed_shape<1>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        packed_softmax<1>(buffer_1_p, buffer_2_p, ntt::fixed_shape<1>{});
    }
    auto t2 = NttTest::get_cpu_cycle();
    unpack<1>(buffer_2_p, buffer_2_up);

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    benchmark_ntt_softmax_no_pack_dim0();
    benchmark_ntt_softmax_no_pack_dim1();
    benchmark_ntt_softmax_pack0_dim0();
    benchmark_ntt_softmax_pack0_dim1();
    benchmark_ntt_softmax_pack1_dim0();
    benchmark_ntt_softmax_pack1_dim1();
    return 0;
}