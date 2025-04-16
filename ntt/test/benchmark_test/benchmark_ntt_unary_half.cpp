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

template <template <typename T> class Op, typename T, size_t N>
void benchmark_ntt_unary_half(std::string op_name, T low, T high) {
#if __riscv
    constexpr size_t size1 = 300;
    constexpr size_t size2 = 600;
#elif __x86_64__
    constexpr size_t size1 = 1000;
    constexpr size_t size2 = 1000;
#else
    constexpr size_t size1 = 2000;
    constexpr size_t size2 = 2000;
#endif
    using tensor_type = ntt::tensor<ntt::vector<T, N>, ntt::fixed_shape<size2>>;
    tensor_type ntt_input, ntt_output;
    NttTest::init_tensor(ntt_input, low, high);

    for (size_t i = 0; i < size1; i++) {
        ntt::unary<Op>(ntt_input, ntt_output);
        asm volatile("" ::"g"(ntt_output));
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < size1; i++) {
        ntt::unary<Op>(ntt_input, ntt_output);
        asm volatile("" ::"g"(ntt_output));
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
    benchmark_ntt_unary_half<ntt::ops::abs, half, N>(
        "abs", half::round_to_half(-10.f), half::round_to_half(10.f));
    benchmark_ntt_unary_half<ntt::ops::acos, half, N>(
        "acos", half::round_to_half(-1.f), half::round_to_half(1.f));
    benchmark_ntt_unary_half<ntt::ops::acosh, half, N>(
        "acosh", half::round_to_half(1.f), half::round_to_half(10.f));
    benchmark_ntt_unary_half<ntt::ops::asin, half, N>(
        "asin", half::round_to_half(-1.f), half::round_to_half(1.f));
    benchmark_ntt_unary_half<ntt::ops::asinh, half, N>(
        "asinh", half::round_to_half(-10.f), half::round_to_half(10.f));
    benchmark_ntt_unary_half<ntt::ops::ceil, half, N>(
        "ceil", half::round_to_half(-10.f), half::round_to_half(10.f));
    benchmark_ntt_unary_half<ntt::ops::copy, half, N>(
        "copy", half::round_to_half(-10.f), half::round_to_half(10.f));
    benchmark_ntt_unary_half<ntt::ops::cos, half, N>(
        "cos", half::round_to_half(-10.f), half::round_to_half(10.f));
    benchmark_ntt_unary_half<ntt::ops::cosh, half, N>(
        "cosh", half::round_to_half(-10.f), half::round_to_half(10.f));
    benchmark_ntt_unary_half<ntt::ops::erf, half, N>(
        "erf", half::round_to_half(-10.f), half::round_to_half(10.f));
    benchmark_ntt_unary_half<ntt::ops::exp, half, N>(
        "exp", half::round_to_half(-10.f), half::round_to_half(10.f));
    benchmark_ntt_unary_half<ntt::ops::floor, half, N>(
        "floor", half::round_to_half(-10.f), half::round_to_half(10.f));
    benchmark_ntt_unary_half<ntt::ops::log, half, N>(
        "log", half::round_to_half(-10.f), half::round_to_half(10.f));
    benchmark_ntt_unary_half<ntt::ops::neg, half, N>(
        "neg", half::round_to_half(-10.f), half::round_to_half(10.f));
    benchmark_ntt_unary_half<ntt::ops::round, half, N>(
        "round", half::round_to_half(-10.f), half::round_to_half(10.f));
    benchmark_ntt_unary_half<ntt::ops::rsqrt, half, N>(
        "rsqrt", half::round_to_half(1.f), half::round_to_half(10.f));
    benchmark_ntt_unary_half<ntt::ops::sign, half, N>(
        "sign", half::round_to_half(-10.f), half::round_to_half(10.f));
    benchmark_ntt_unary_half<ntt::ops::sin, half, N>(
        "sin", half::round_to_half(-10.f), half::round_to_half(10.f));
    benchmark_ntt_unary_half<ntt::ops::sinh, half, N>(
        "sinh", half::round_to_half(-10.f), half::round_to_half(10.f));
    benchmark_ntt_unary_half<ntt::ops::sqrt, half, N>(
        "sqrt", half::round_to_half(1.f), half::round_to_half(10.f));
    benchmark_ntt_unary_half<ntt::ops::square, half, N>(
        "square", half::round_to_half(-10.f), half::round_to_half(10.f));
    benchmark_ntt_unary_half<ntt::ops::swish, half, N>(
        "swish", half::round_to_half(-10.f), half::round_to_half(10.f));
    benchmark_ntt_unary_half<ntt::ops::tanh, half, N>(
        "tanh", half::round_to_half(-10.f), half::round_to_half(10.f));
}