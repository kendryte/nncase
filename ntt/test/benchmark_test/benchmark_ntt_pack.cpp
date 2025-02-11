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

template <typename ElementType, size_t N, size_t C, size_t H, size_t W,
          size_t... pack_dims>
void benchmark_ntt_pack(const std::string &mode, const size_t run_size) {
    constexpr size_t axes[] = {pack_dims...};
    constexpr size_t axes_size = sizeof...(pack_dims);
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t P0 =
        std::any_of(axes, axes + axes_size, [](size_t i) { return i == 0; })
            ? P
            : 1;
    constexpr size_t P1 =
        std::any_of(axes, axes + axes_size, [](size_t i) { return i == 1; })
            ? P
            : 1;
    constexpr size_t P2 =
        std::any_of(axes, axes + axes_size, [](size_t i) { return i == 2; })
            ? P
            : 1;
    constexpr size_t P3 =
        std::any_of(axes, axes + axes_size, [](size_t i) { return i == 3; })
            ? P
            : 1;
    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<N, C, H, W>>;
    using tensor_type2 =
        ntt::tensor<ElementType,
                    ntt::fixed_shape<N / P0, C / P1, H / P2, W / P3>>;

    alignas(32) tensor_type1 ntt_input;
    alignas(32) tensor_type2 ntt_output;
    NttTest::init_tensor(ntt_input, -10.f, 10.f);

    // warm up
    constexpr size_t warmup_size = 10;
    for (size_t i = 0; i < warmup_size; i++) {
        ntt::pack<pack_dims...>(ntt_input, ntt_output);
        asm volatile("" ::"g"(ntt_output));
    }

    // run
    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_size; i++) {
        ntt::pack<pack_dims...>(ntt_input, ntt_output);
        asm volatile("" ::"g"(ntt_output));
    }
    auto t2 = NttTest::get_cpu_cycle();

    auto element_size = tensor_type2::size() * ElementType::size() / P;
    std::cout << __FUNCTION__ << "_" << mode << " took " << std::setprecision(1)
              << std::fixed
              << static_cast<float>(t2 - t1) / run_size / element_size
              << " cycles" << std::endl;
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

#if __riscv
    benchmark_ntt_pack<ntt::vector<float, P>, 16 * P, 3, 4, 4, 0>("N", 300);
    benchmark_ntt_pack<ntt::vector<float, P>, 3, 16 * P, 4, 4, 1>("C", 300);
    benchmark_ntt_pack<ntt::vector<float, P>, 3, 4, 16 * P, 4, 2>("H", 300);
    benchmark_ntt_pack<ntt::vector<float, P>, 3, 4, 4, 16 * P, 3>("W", 300);
    benchmark_ntt_pack<ntt::vector<float, P, P>, 4 * P, 3 * P, 4, 4, 0, 1>("NC",
                                                                           300);
    benchmark_ntt_pack<ntt::vector<float, P, P>, 2, 3 * P, 4 * P, 8, 1, 2>("CH",
                                                                           300);
    benchmark_ntt_pack<ntt::vector<float, P, P>, 4, 4, 3 * P, 4 * P, 2, 3>("HW",
                                                                           300);
#elif __x86_64__
    benchmark_ntt_pack<ntt::vector<float, P>, P * 8, 2, 2, 2, 0>("N", 2000);
    benchmark_ntt_pack<ntt::vector<float, P>, 2, 8 * P, 2, 4, 1>("C", 2000);
    benchmark_ntt_pack<ntt::vector<float, P>, 2, 2, 8 * P, 8, 2>("H", 2000);
    benchmark_ntt_pack<ntt::vector<float, P>, 2, 2, 2, 8 * P, 3>("W", 2000);
    benchmark_ntt_pack<ntt::vector<float, P, P>, 4 * P, 8 * P, 2, 4, 0, 1>("NC",
                                                                           1);
    benchmark_ntt_pack<ntt::vector<float, P, P>, 2, 4 * P, 8 * P, 8, 1, 2>(
        "CH", 2000);
    benchmark_ntt_pack<ntt::vector<float, P, P>, 4, 4, 8 * P, 8 * P, 2, 3>(
        "HW", 2000);
#else
    benchmark_ntt_pack<ntt::vector<float, P>, 16 * P, 3, 4, 4, 0>("N", 300);
    benchmark_ntt_pack<ntt::vector<float, P>, 3, 16 * P, 4, 4, 1>("C", 300);
    benchmark_ntt_pack<ntt::vector<float, P>, 3, 4, 16 * P, 4, 2>("H", 300);
    benchmark_ntt_pack<ntt::vector<float, P>, 3, 4, 4, 16 * P, 3>("W", 300);
    benchmark_ntt_pack<ntt::vector<float, P, P>, 4 * P, 3 * P, 4, 4, 0, 1>("NC",
                                                                           300);
    benchmark_ntt_pack<ntt::vector<float, P, P>, 2, 3 * P, 4 * P, 8, 1, 2>("CH",
                                                                           300);
    benchmark_ntt_pack<ntt::vector<float, P, P>, 4, 4, 3 * P, 4 * P, 2, 3>("HW",
                                                                           300);
#endif
}