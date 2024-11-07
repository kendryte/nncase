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
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <memory>
#include <nncase/ntt/ntt.h>
#include <stdexcept>
#include <string>

using namespace nncase;

template <size_t... PackDims> std::string pack_pattern() {
    if constexpr (sizeof...(PackDims) == 1) {
        constexpr size_t pack_dims[] = {PackDims...};
        if constexpr (pack_dims[0] == 0) {
            return "pack1d_dim0";
        } else if constexpr (pack_dims[0] == 1) {
            return "pack1d_dim1";
        } else {
            std::cerr << "Invalid PackDims" << std::endl;
            std::abort();
        }
    } else if constexpr (sizeof...(PackDims) == 2) {
        return "pack2d";
    } else {
        std::cerr << "Invalid PackDims for pack2d" << std::endl;
        std::abort();
    }
};

template <size_t... PackDims> void benchmark_ntt_pack() {
    constexpr size_t pack_dims[] = {PackDims...};
    constexpr size_t NumDims = sizeof...(PackDims);

    if constexpr (sizeof...(PackDims) != 2 && sizeof...(PackDims) != 1) {
        std::cerr << "unsupported data type" << std::endl;
        std::abort();
    }

    auto pattern = pack_pattern<PackDims...>();

    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t P0 =
        pack_dims[0] == 0 ? NTT_VLEN / (sizeof(float) * 8) : 1;
    constexpr size_t P1 =
        [=] {
            if constexpr (sizeof...(PackDims) == 2)
                return pack_dims[1] == 1;
            else
                return pack_dims[0] == 1;
        }()
            ? NTT_VLEN / (sizeof(float) * 8)
            : 1;

    using ElementType =
        std::conditional_t<NumDims == 2, ntt::vector<float, P0, P1>,
                           ntt::vector<float, P>>;

    // pay attention to the following code
    constexpr size_t warmup_size = 10;
#if __riscv
    constexpr size_t M = 12 * P;
    constexpr size_t N = 16 * P;
    constexpr size_t run_size = 300;
#elif __x86_64__
    constexpr size_t M = 8 * P;
    constexpr size_t N = 8 * P;
    constexpr size_t run_size = 2000;
#else
    constexpr size_t M = 12 * P;
    constexpr size_t N = 16 * P;
    constexpr size_t run_size = 2000;
#endif

    using tensor_a_type = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    using tensor_b_type =
        ntt::tensor<ElementType, ntt::fixed_shape<M / P0, N / P1>>;

    tensor_a_type ta;
    tensor_b_type tb;
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);

    // warm up
    for (size_t i = 0; i < warmup_size; i++)
        ntt::pack<PackDims...>(ta, tb);

    // run
    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_size; i++) {
        ntt::pack<PackDims...>(ta, tb);
        asm volatile("" ::"g"(tb));
    }
    auto t2 = NttTest::get_cpu_cycle();

    auto element_size = tensor_b_type::size() * ElementType::size() / P;
    std::cout << __FUNCTION__ << "_" << pattern << " took "
              << std::setprecision(1) << std::fixed
              << static_cast<float>(t2 - t1) / run_size / element_size
              << " cycles" << std::endl;
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    benchmark_ntt_pack<0>();
    benchmark_ntt_pack<1>();
    benchmark_ntt_pack<0, 1>();
}