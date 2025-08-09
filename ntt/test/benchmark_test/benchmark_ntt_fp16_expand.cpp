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

template <size_t N, size_t C, size_t H, size_t W, size_t expand_n,
          size_t expand_c, size_t expand_h, size_t expand_w>
void benchmark_ntt_expand(const std::string &mode) {

    constexpr size_t run_size = 2000;
    constexpr size_t P = NTT_VLEN / (sizeof(half) * 8);

    auto ntt_input = ntt::make_tensor<half>(ntt::fixed_shape_v<N, C, H, W>);
    auto ntt_output = ntt::make_tensor<half>(
        ntt::fixed_shape_v<expand_n, expand_c, expand_h, expand_w>);
    std::iota(ntt_input.elements().begin(), ntt_input.elements().end(), 0.f);

    // warm up
    constexpr size_t warmup_size = 10;
    for (size_t i = 0; i < warmup_size; i++) {
        ntt::expand(ntt_input, ntt_output);
        asm volatile("" ::"g"(ntt_output));
    }

    // run
    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_size; i++) {
        ntt::expand(ntt_input, ntt_output);
        asm volatile("" ::"g"(ntt_output));
    }
    auto t2 = NttTest::get_cpu_cycle();

    auto element_size = ntt_output.size() / P;
    std::cout << __FUNCTION__ << "_" << mode << " took " << std::setprecision(1)
              << std::fixed
              << static_cast<float>(t2 - t1) / run_size / element_size
              << " cycles" << std::endl;
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    benchmark_ntt_expand<32, 32, 32, 1, 32, 32, 32, 32>("W");
    benchmark_ntt_expand<32, 32, 1, 32, 32, 32, 32, 32>("H");
    benchmark_ntt_expand<32, 1, 32, 32, 32, 32, 32, 32>("C");
    benchmark_ntt_expand<1, 16, 16, 16, 16, 16, 16, 16>("N");
    benchmark_ntt_expand<1, 1, 32, 32, 32, 32, 32, 32>("NC");
    benchmark_ntt_expand<32, 1, 1, 32, 32, 32, 32, 32>("CH");
    benchmark_ntt_expand<32, 32, 1, 1, 32, 32, 32, 32>("HW");
    benchmark_ntt_expand<1, 32, 1, 32, 32, 32, 32, 32>("NH");
    benchmark_ntt_expand<32, 1, 32, 1, 32, 32, 32, 32>("CW");
    benchmark_ntt_expand<1, 32, 32, 1, 32, 32, 32, 32>("NW");
}