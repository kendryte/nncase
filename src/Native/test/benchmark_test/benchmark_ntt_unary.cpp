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

template <template <typename T1> class Op, typename T2, size_t N>
void benchmark_ntt_unary(std::string op_name, T2 low, T2 high) {
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
    using tensor_type =
        ntt::tensor<ntt::vector<T2, N>, ntt::fixed_shape<size2>>;
    tensor_type ntt_input, ntt_result;
    NttTest::init_tensor(ntt_input, low, high);

    Op<tensor_type> op;
    for (size_t i = 0; i < size1; i++)
        ntt_result = op(ntt_input);
    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < size1; i++)
        ntt_result = op(ntt_input);
    auto t2 = NttTest::get_cpu_cycle();
#if __x86_64__
    asm volatile("" ::"g"(ntt_result));
#endif
    std::cout << __FUNCTION__ << "_" << op_name << " took "
              << std::setprecision(1) << std::fixed
              << static_cast<float>(t2 - t1) / size1 / size2 << " cycles"
              << std::endl;
}

#define BENCHMARK_NTT_UNARY(OP)                                                \
    template <size_t N, size_t run_size, size_t size>                          \
    void benchmark_ntt_unary_##OP() {                                          \
                                                                               \
        using tensor_type1 =                                                   \
            ntt::tensor<ntt::vector<float, N>, ntt::fixed_shape<size>>;        \
        using tensor_type2 =                                                   \
            ntt::tensor<ntt::vector<float, N>, ntt::fixed_shape<size>>;        \
        constexpr size_t warmup_size = 30;                                     \
                                                                               \
        tensor_type1 ntt_input;                                                \
        tensor_type2 ntt_output;                                               \
        NttTest::init_tensor(ntt_input, -10.f, 10.f);                          \
                                                                               \
        for (size_t i = 0; i < warmup_size; i++)                               \
            ntt::OP(ntt_input, ntt_output);                                    \
                                                                               \
        auto t1 = NttTest::get_cpu_cycle();                                    \
        for (size_t i = 0; i < run_size; i++) {                                \
            ntt::OP(ntt_input, ntt_output);                                    \
            asm volatile("" ::"g"(ntt_output));                                \
            asm volatile("" ::"g"(ntt_input));                                 \
        }                                                                      \
        auto t2 = NttTest::get_cpu_cycle();                                    \
                                                                               \
        std::cout << __FUNCTION__ << " took " << std::setprecision(1)          \
                  << std::fixed                                                \
                  << static_cast<float>(t2 - t1) / size / run_size             \
                  << " cycles" << std::endl;                                   \
    }

BENCHMARK_NTT_UNARY(ceil)
BENCHMARK_NTT_UNARY(abs)
BENCHMARK_NTT_UNARY(floor)
BENCHMARK_NTT_UNARY(neg)
BENCHMARK_NTT_UNARY(round)
BENCHMARK_NTT_UNARY(sign)
BENCHMARK_NTT_UNARY(square)

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

#if __riscv
    constexpr size_t run_size = 300;
    constexpr size_t size = 600;
#elif __x86_64__
    constexpr size_t run_size = 2000;
    constexpr size_t size = 2000;
#else
    constexpr size_t run_size = 2000;
    constexpr size_t size = 2000;
#endif
    constexpr size_t N = NTT_VLEN / (sizeof(float) * 8);
    // benchmark_ntt_unary<ntt::ops::abs, float, N>("abs", -10.f, 10.f);
    benchmark_ntt_unary<ntt::ops::acos, float, N>("acos", -1.f, 1.f);
    benchmark_ntt_unary<ntt::ops::acosh, float, N>("acosh", 1.f, 10.f);
    benchmark_ntt_unary<ntt::ops::asin, float, N>("asin", -1.f, 1.f);
    benchmark_ntt_unary<ntt::ops::asinh, float, N>("asinh", -10.f, 10.f);
    // benchmark_ntt_unary<ntt::ops::ceil, float, N>("ceil", -10.f, 10.f);
    benchmark_ntt_unary<ntt::ops::cos, float, N>("cos", -10.f, 10.f);
    benchmark_ntt_unary<ntt::ops::cosh, float, N>("cosh", -10.f, 10.f);
    benchmark_ntt_unary<ntt::ops::erf, float, N>("erf", -10.f, 10.f);
    benchmark_ntt_unary<ntt::ops::exp, float, N>("exp", -10.f, 10.f);
    // benchmark_ntt_unary<ntt::ops::floor, float, N>("floor", -10.f, 10.f);
    benchmark_ntt_unary<ntt::ops::log, float, N>("log", -10.f, 10.f);
    // benchmark_ntt_unary<ntt::ops::neg, float, N>("neg", -10.f, 10.f);
    // benchmark_ntt_unary<ntt::ops::round, float, N>("round", -10.f, 10.f);
    benchmark_ntt_unary<ntt::ops::rsqrt, float, N>("rsqrt", 1.f, 10.f);
    // benchmark_ntt_unary<ntt::ops::sign, float, N>("sign", -10.f, 10.f);
    benchmark_ntt_unary<ntt::ops::sin, float, N>("sin", -10.f, 10.f);
    benchmark_ntt_unary<ntt::ops::sinh, float, N>("sinh", -10.f, 10.f);
    benchmark_ntt_unary<ntt::ops::sqrt, float, N>("sqrt", 1.f, 10.f);
    // benchmark_ntt_unary<ntt::ops::square, float, N>("square", -10.f, 10.f);
    benchmark_ntt_unary<ntt::ops::swish, float, N>("swish", -10.f, 10.f);
    benchmark_ntt_unary<ntt::ops::tanh, float, N>("tanh", -10.f, 10.f);
    benchmark_ntt_unary_ceil<N, run_size, size>();
    benchmark_ntt_unary_abs<N, run_size, size>();
    benchmark_ntt_unary_floor<N, run_size, size>();
    benchmark_ntt_unary_neg<N, run_size, size>();
    benchmark_ntt_unary_round<N, run_size, size>();
    benchmark_ntt_unary_sign<N, run_size, size>();
    benchmark_ntt_unary_square<N, run_size, size>();
}