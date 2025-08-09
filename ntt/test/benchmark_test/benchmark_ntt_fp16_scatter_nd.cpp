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
void benchmark_ntt_scatterND_unpack(T init_low, T init_high, int64_t idx0,
                                    int64_t idx1, int64_t idx2, int64_t idx3,
                                    int64_t idx4) {
    // #if __riscv
    //     constexpr size_t size1 = 300;
    //     constexpr size_t size2 = 600;
    // #elif __x86_64__
    //     constexpr size_t size1 = 2000;
    //     constexpr size_t size2 = 2000;
    // #else
    //     constexpr size_t size1 = 2000;
    //     constexpr size_t size2 = 2000;
    // #endif

    auto shape1 = ntt::fixed_shape_v<10, 64, 64, 32>;
    auto ntt_output = ntt::make_unique_tensor<T>(shape1);

    // Initialize *input with random values
    auto input = ntt::make_unique_tensor<T>(shape1);
    NttTest::init_tensor(*input, init_low, init_high);

    // Initialize *updates with random values
    auto shape2 = ntt::fixed_shape_v<5, 64, 64, 32>;
    auto updates = ntt::make_unique_tensor<T>(shape2);
    NttTest::init_tensor(*updates, init_low, init_high);

    // Initialize indices
    auto indices = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<5, 1>);
    indices(0, 0) = idx0;
    indices(1, 0) = idx1;
    indices(2, 0) = idx2;
    indices(3, 0) = idx3;
    indices(4, 0) = idx4;

    // warm up
    constexpr size_t warmup_size = 10;
    for (size_t i = 0; i < warmup_size; i++) {
        ntt::scatter_nd(*input, indices, *updates, *ntt_output);
        asm volatile("" ::"g"(ntt_output));
    }

    // run
    constexpr size_t run_size = 2000;
    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_size; i++) {
        ntt::scatter_nd(*input, indices, *updates, *ntt_output);
        asm volatile("" ::"g"(ntt_output));
    }
    auto t2 = NttTest::get_cpu_cycle();

    auto element_size = updates->size();
    std::cout << __FUNCTION__ << " took " << std::setprecision(1) << std::fixed
              << static_cast<float>(t2 - t1) / run_size / element_size
              << " cycles" << std::endl;
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    constexpr size_t N = NTT_VLEN / (sizeof(half) * 8);
    benchmark_ntt_scatterND_unpack<half, N>(half(-10.f), half(10.f), 1, 3, 5, 7,
                                            9);
}