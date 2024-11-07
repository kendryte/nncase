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
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <memory>
#include <nncase/ntt/ntt.h>
#include <stdexcept>
#include <string>

using namespace nncase;

template <size_t... unpack_dims> void benchmark_ntt_unpack(const std::string &mode) {
    constexpr size_t dims[] = {unpack_dims...};
    constexpr size_t ndims = sizeof...(unpack_dims);

    if constexpr (ndims != 2 && ndims != 1) {
        std::cerr << "unsupported unpack dims" << std::endl;
        std::abort();
    }

    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t P0 = dims[0] == 0 ? P : 1;
    constexpr size_t P1 =
        [=] {
            if constexpr (ndims == 2)
                return dims[1] == 1;
            else
                return dims[0] == 1;
        }()
            ? P
            : 1;

    using ElementType =
        std::conditional_t<ndims == 2, ntt::vector<float, P0, P1>,
                           ntt::vector<float, P>>;

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

    using tensor_type1 =
        ntt::tensor<ElementType, ntt::fixed_shape<M / P0, N / P1>>;
    using tensor_type2 = ntt::tensor<float, ntt::fixed_shape<M, N>>;

    tensor_type1 ntt_input;
    tensor_type2 ntt_output;
    NttTest::init_tensor(ntt_input, -10.f, 10.f);

    // warm up
    constexpr size_t warmup_size = 10;
    for (size_t i = 0; i < warmup_size; i++)
    {
        ntt::unpack<unpack_dims...>(ntt_input, ntt_output);
        asm volatile("" ::"g"(ntt_output));
    }

    // run
    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_size; i++) {
        ntt::unpack<unpack_dims...>(ntt_input, ntt_output);
        asm volatile("" ::"g"(ntt_output));
    }
    auto t2 = NttTest::get_cpu_cycle();

    auto element_size = tensor_type1::size() * ElementType::size() / P;
    std::cout << __FUNCTION__ << "_" << mode << " took "
              << std::setprecision(1) << std::fixed
              << static_cast<float>(t2 - t1) / run_size / element_size
              << " cycles" << std::endl;
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    benchmark_ntt_unpack<0>("pack1d_dim0");
    benchmark_ntt_unpack<1>("pack1d_dim1");
    benchmark_ntt_unpack<0, 1>("pack2d");
}