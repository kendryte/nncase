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

template <size_t perm_n, size_t perm_c, size_t perm_h, size_t perm_w>
void benchmark_ntt_transpose(const std::string &mode) {
#if __riscv
    constexpr size_t n = 4;
    constexpr size_t c = 5;
    constexpr size_t h = 6;
    constexpr size_t w = 8;
#elif __x86_64__
    constexpr size_t n = 6;
    constexpr size_t c = 6;
    constexpr size_t h = 6;
    constexpr size_t w = 6;
#else
    constexpr size_t n = 4;
    constexpr size_t c = 5;
    constexpr size_t h = 6;
    constexpr size_t w = 8;
#endif
    constexpr size_t run_size = 2000;
    constexpr size_t P = NTT_VLEN / (sizeof(half) * 8);

    constexpr std::array<size_t, 4> org_dims = {n, c, h, w};
    constexpr std::array<size_t, 4> new_dims = {
        org_dims[perm_n], org_dims[perm_c], org_dims[perm_h], org_dims[perm_w]};

    auto ntt_input =
        ntt::make_tensor<ntt::vector<half, P>>(ntt::fixed_shape_v<n, c, h, w>);
    auto ntt_output = ntt::make_tensor<ntt::vector<half, P>>(
        ntt::fixed_shape_v<new_dims[0], new_dims[1], new_dims[2], new_dims[3]>);
    NttTest::init_tensor(ntt_input, half(-10.f), half(10.f));

    // warm up
    constexpr size_t warmup_size = 10;
    for (size_t i = 0; i < warmup_size; i++) {
        ntt::transpose(ntt_input, ntt_output,
                       ntt::fixed_shape_v<perm_n, perm_c, perm_h, perm_w>);
        asm volatile("" ::"g"(ntt_output));
    }

    // run
    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_size; i++) {
        ntt::transpose(ntt_input, ntt_output,
                       ntt::fixed_shape_v<perm_n, perm_c, perm_h, perm_w>);
        asm volatile("" ::"g"(ntt_output));
    }
    auto t2 = NttTest::get_cpu_cycle();

    // run
    t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_size; i++) {
        ntt::transpose(ntt_input, ntt_output,
                       ntt::fixed_shape_v<perm_n, perm_c, perm_h, perm_w>);
        asm volatile("" ::"g"(ntt_output));
    }
    t2 = NttTest::get_cpu_cycle();

    auto element_size = ntt_output.size();
    std::cout << __FUNCTION__ << "_" << mode << " took " << std::setprecision(1)
              << std::fixed
              << static_cast<float>(t2 - t1) / run_size / element_size
              << " cycles" << std::endl;
}

template <size_t perm_n, size_t perm_c, size_t perm_h, size_t perm_w>
void benchmark_ntt_transpose_warmup([[maybe_unused]] const std::string &mode) {
#if __riscv
    constexpr size_t n = 4;
    constexpr size_t c = 5;
    constexpr size_t h = 6;
    constexpr size_t w = 8;
#elif __x86_64__
    constexpr size_t n = 6;
    constexpr size_t c = 6;
    constexpr size_t h = 6;
    constexpr size_t w = 6;
#else
    constexpr size_t n = 4;
    constexpr size_t c = 5;
    constexpr size_t h = 6;
    constexpr size_t w = 8;
#endif
    constexpr size_t run_size = 2000;
    constexpr size_t P = NTT_VLEN / (sizeof(half) * 8);

    constexpr std::array<size_t, 4> org_dims = {n, c, h, w};
    constexpr std::array<size_t, 4> new_dims = {
        org_dims[perm_n], org_dims[perm_c], org_dims[perm_h], org_dims[perm_w]};

    auto ntt_input =
        ntt::make_tensor<ntt::vector<half, P>>(ntt::fixed_shape_v<n, c, h, w>);
    auto ntt_output = ntt::make_tensor<ntt::vector<half, P>>(
        ntt::fixed_shape_v<new_dims[0], new_dims[1], new_dims[2], new_dims[3]>);
    NttTest::init_tensor(ntt_input, half(-10.f), half(10.f));

    // warm up
    constexpr size_t warmup_size = 10;
    for (size_t i = 0; i < warmup_size; i++) {
        ntt::transpose(ntt_input, ntt_output,
                       ntt::fixed_shape_v<perm_n, perm_c, perm_h, perm_w>);
        asm volatile("" ::"g"(ntt_output));
    }

    // run
    [[maybe_unused]] auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_size; i++) {
        ntt::transpose(ntt_input, ntt_output,
                       ntt::fixed_shape_v<perm_n, perm_c, perm_h, perm_w>);
        asm volatile("" ::"g"(ntt_output));
    }
    [[maybe_unused]] auto t2 = NttTest::get_cpu_cycle();
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    benchmark_ntt_transpose_warmup<0, 1, 2, 3>("NCHW");
    benchmark_ntt_transpose<0, 1, 2, 3>("NCHW");
    benchmark_ntt_transpose<0, 1, 3, 2>("NCWH");
    benchmark_ntt_transpose<0, 2, 1, 3>("NHCW");
    benchmark_ntt_transpose<0, 2, 3, 1>("NHWC");
    benchmark_ntt_transpose<0, 3, 1, 2>("NWCH");
    benchmark_ntt_transpose<0, 3, 2, 1>("NWHC");
    benchmark_ntt_transpose<1, 0, 2, 3>("CNHW");
    benchmark_ntt_transpose<1, 0, 3, 2>("CNWH");
    benchmark_ntt_transpose<1, 2, 0, 3>("CHNW");
    benchmark_ntt_transpose<1, 2, 3, 0>("CHWN");
    benchmark_ntt_transpose<1, 3, 0, 2>("CWNH");
    benchmark_ntt_transpose<1, 3, 2, 0>("CWHN");
    benchmark_ntt_transpose<2, 0, 1, 3>("HNCW");
    benchmark_ntt_transpose<2, 0, 3, 1>("HNWC");
    benchmark_ntt_transpose<2, 1, 0, 3>("HCNW");
    benchmark_ntt_transpose<2, 1, 3, 0>("HCWN");
    benchmark_ntt_transpose<2, 3, 0, 1>("HWNC");
    benchmark_ntt_transpose<2, 3, 1, 0>("HWCN");
    benchmark_ntt_transpose<3, 0, 1, 2>("WNCH");
    benchmark_ntt_transpose<3, 0, 2, 1>("WNHC");
    benchmark_ntt_transpose<3, 1, 0, 2>("WCNH");
    benchmark_ntt_transpose<3, 1, 2, 0>("WCHN");
    benchmark_ntt_transpose<3, 2, 0, 1>("WHNC");
    benchmark_ntt_transpose<3, 2, 1, 0>("WHCN");
}