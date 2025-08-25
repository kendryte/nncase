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
void benchmark_ntt_unary(std::string op_name, T low, T high) {
#if __riscv
    constexpr size_t size1 = 300;
    constexpr size_t size2 = 600;
#elif __x86_64__
    constexpr size_t size1 = 1;
    constexpr size_t size2 = 100;
#else
    constexpr size_t size1 = 2000;
    constexpr size_t size2 = 2000;
#endif
    auto ntt_input =
        ntt::make_tensor<ntt::vector<T, N>>(ntt::fixed_shape_v<size2>);
    auto ntt_output =
        ntt::make_tensor<ntt::vector<T, N>>(ntt::fixed_shape_v<size2>);
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

    constexpr size_t N = NTT_VLEN / (sizeof(bfloat16) * 8);
    benchmark_ntt_unary<ntt::ops::abs, bfloat16, N>("abs", bfloat16(-10.f),
                                                    bfloat16(10.f));
    benchmark_ntt_unary<ntt::ops::acos, bfloat16, N>("acos", bfloat16(-1.f),
                                                     bfloat16(1.f));
    benchmark_ntt_unary<ntt::ops::acosh, bfloat16, N>("acosh", bfloat16(1.f),
                                                      bfloat16(10.f));
    benchmark_ntt_unary<ntt::ops::asin, bfloat16, N>("asin", bfloat16(-1.f),
                                                     bfloat16(1.f));
    benchmark_ntt_unary<ntt::ops::asinh, bfloat16, N>("asinh", bfloat16(-10.f),
                                                      bfloat16(10.f));
    benchmark_ntt_unary<ntt::ops::ceil, bfloat16, N>("ceil", bfloat16(-10.f),
                                                     bfloat16(10.f));
    benchmark_ntt_unary<ntt::ops::copy, bfloat16, N>("copy", bfloat16(-10.f),
                                                     bfloat16(10.f));
    benchmark_ntt_unary<ntt::ops::cos, bfloat16, N>("cos", bfloat16(-10.f),
                                                    bfloat16(10.f));
    benchmark_ntt_unary<ntt::ops::cosh, bfloat16, N>("cosh", bfloat16(-10.f),
                                                     bfloat16(10.f));
    benchmark_ntt_unary<ntt::ops::erf, bfloat16, N>("erf", bfloat16(-10.f),
                                                    bfloat16(10.f));
    benchmark_ntt_unary<ntt::ops::exp, bfloat16, N>("exp", bfloat16(-10.f),
                                                    bfloat16(10.f));
    benchmark_ntt_unary<ntt::ops::floor, bfloat16, N>("floor", bfloat16(-10.f),
                                                      bfloat16(10.f));
    benchmark_ntt_unary<ntt::ops::log, bfloat16, N>("log", bfloat16(-10.f),
                                                    bfloat16(10.f));
    benchmark_ntt_unary<ntt::ops::neg, bfloat16, N>("neg", bfloat16(-10.f),
                                                    bfloat16(10.f));
    benchmark_ntt_unary<ntt::ops::round, bfloat16, N>("round", bfloat16(-10.f),
                                                      bfloat16(10.f));
    benchmark_ntt_unary<ntt::ops::rsqrt, bfloat16, N>("rsqrt", bfloat16(1.f),
                                                      bfloat16(10.f));
    benchmark_ntt_unary<ntt::ops::sign, bfloat16, N>("sign", bfloat16(-10.f),
                                                     bfloat16(10.f));
    benchmark_ntt_unary<ntt::ops::sin, bfloat16, N>("sin", bfloat16(-10.f),
                                                    bfloat16(10.f));
    benchmark_ntt_unary<ntt::ops::sinh, bfloat16, N>("sinh", bfloat16(-10.f),
                                                     bfloat16(10.f));
    benchmark_ntt_unary<ntt::ops::sqrt, bfloat16, N>("sqrt", bfloat16(1.f),
                                                     bfloat16(10.f));
    benchmark_ntt_unary<ntt::ops::square, bfloat16, N>(
        "square", bfloat16(-10.f), bfloat16(10.f));
    benchmark_ntt_unary<ntt::ops::swish, bfloat16, N>("swish", bfloat16(-10.f),
                                                      bfloat16(10.f));
    benchmark_ntt_unary<ntt::ops::tanh, bfloat16, N>("tanh", bfloat16(-10.f),
                                                     bfloat16(10.f));
}