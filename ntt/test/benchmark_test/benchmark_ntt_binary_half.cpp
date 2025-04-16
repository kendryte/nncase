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
#include "nncase/half.h"
#include <iomanip>
#include <nncase/ntt/ntt.h>

using namespace nncase;

template <template <typename T1, typename T2> class Op, typename T, size_t N>
void benchmark_ntt_binary_half(std::string op_name, T lhs_low, T lhs_high, T rhs_low,
                          T rhs_high) {
#if __riscv
    constexpr size_t size1 = 300;
    constexpr size_t size2 = 600;
#elif __x86_64__
    constexpr size_t size1 = 2000;
    constexpr size_t size2 = 2000;
#else
    constexpr size_t size1 = 2000;
    constexpr size_t size2 = 2000;
#endif
    using tensor_type = ntt::tensor<ntt::vector<T, N>, ntt::fixed_shape<size2>>;
    tensor_type ntt_lhs, ntt_rhs, ntt_result;
    NttTest::init_tensor(ntt_lhs, lhs_low, lhs_high);
    NttTest::init_tensor(ntt_rhs, rhs_low, rhs_high);

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < size1; i++) {
        ntt::binary<Op>(ntt_lhs, ntt_rhs, ntt_result);
        asm volatile("" ::"g"(ntt_result));
    }

    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << "_" << op_name << " took "
              << std::setprecision(1) << std::fixed
              << static_cast<float>(t2 - t1) / size1 / size2 << " cycles"
              << std::endl;
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    constexpr size_t N = NTT_VLEN / (sizeof(half) * 8);
    benchmark_ntt_binary_half<ntt::ops::add, half, N>(
        "add", half::round_to_half(-10.f), half::round_to_half(10.f),
        half::round_to_half(-10.f), half::round_to_half(10.f));
    benchmark_ntt_binary_half<ntt::ops::sub, half, N>(
        "sub", half::round_to_half(-10.f), half::round_to_half(10.f),
        half::round_to_half(-10.f), half::round_to_half(10.f));
    benchmark_ntt_binary_half<ntt::ops::mul, half, N>(
        "mul", half::round_to_half(-10.f), half::round_to_half(10.f),
        half::round_to_half(-10.f), half::round_to_half(10.f));
    benchmark_ntt_binary_half<ntt::ops::div, half, N>(
        "div", half::round_to_half(-10.f), half::round_to_half(10.f),
        half::round_to_half(1.f), half::round_to_half(10.f));
    benchmark_ntt_binary_half<ntt::ops::max, half, N>(
        "max", half::round_to_half(-10.f), half::round_to_half(10.f),
        half::round_to_half(-10.f), half::round_to_half(10.f));
    benchmark_ntt_binary_half<ntt::ops::min, half, N>(
        "min", half::round_to_half(-10.f), half::round_to_half(10.f),
        half::round_to_half(-10.f), half::round_to_half(10.f));
    benchmark_ntt_binary_half<ntt::ops::floor_mod, int16_t, N>("floor_mod", -10,
                                                               10, 1, 10);
    benchmark_ntt_binary_half<ntt::ops::mod, half, N>(
        "mod", half::round_to_half(-10.f), half::round_to_half(10.f),
        half::round_to_half(1.f), half::round_to_half(10.f));
    benchmark_ntt_binary_half<ntt::ops::pow, half, N>(
        "pow", half::round_to_half(0.f), half::round_to_half(3.f),
        half::round_to_half(0.f), half::round_to_half(3.f));
}