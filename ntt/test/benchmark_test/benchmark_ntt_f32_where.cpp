/* Copyright 2019-2024 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obntt_inputin a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limintt_inputtions under the License.
 */
#include "ntt_test.h"
#include <algorithm>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <memory>
#include <nncase/ntt/ntt.h>
#include <stdexcept>
#include <string>

using namespace nncase;

template <typename T, size_t N>
void benchmark_ntt_where_pack(T init_low, T init_high) {
#ifdef __riscv
    constexpr size_t n = 1;
    constexpr size_t c = 8;
    constexpr size_t h = 16;
    constexpr size_t w = 32;
#elif __x86_64__
    constexpr size_t n = 1;
    constexpr size_t c = 8;
    constexpr size_t h = 16;
    constexpr size_t w = 16;
#else
    constexpr size_t n = 1;
    constexpr size_t c = 8;
    constexpr size_t h = 16;
    constexpr size_t w = 32;
#endif

    // Initialize input with random values
    auto input1 =
        ntt::make_tensor<ntt::vector<T, N>>(ntt::fixed_shape_v<n, c, h, w / N>);
    NttTest::init_tensor(input1, init_low, init_high);
    auto input2 =
        ntt::make_tensor<ntt::vector<T, N>>(ntt::fixed_shape_v<n, c, h, w / N>);
    NttTest::init_tensor(input2, init_low, init_high);
    auto cond = ntt::make_tensor<bool>(ntt::fixed_shape_v<n, c, h, w>);
    NttTest::init_tensor(cond, 0, 1);
    auto cond_packed = ntt::make_tensor<ntt::vector<bool, N>>(
        ntt::fixed_shape_v<n, c, h, w / N>);
    ntt::pack(cond, cond_packed, ntt::fixed_shape_v<3>);

    auto ntt_output =
        ntt::make_tensor<ntt::vector<T, N>>(ntt::fixed_shape_v<n, c, h, w / N>);

    // warm up
    constexpr size_t warmup_size = 10;
    for (size_t i = 0; i < warmup_size; i++) {
        ntt::where(cond_packed, input1, input2, ntt_output);
        asm volatile("" ::"g"(ntt_output));
    }

    // run
    constexpr size_t run_size = 2000;
    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_size; i++) {
        ntt::where(cond_packed, input1, input2, ntt_output);
        asm volatile("" ::"g"(ntt_output));
    }
    auto t2 = NttTest::get_cpu_cycle();

    auto element_size = input1.size();
    std::cout << __FUNCTION__ << " took " << std::setprecision(1) << std::fixed
              << static_cast<float>(t2 - t1) / run_size / element_size
              << " cycles" << std::endl;
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    constexpr size_t N = NTT_VLEN / (sizeof(float) * 8);
    benchmark_ntt_where_pack<float, N>(-10.f, 10.f);
}