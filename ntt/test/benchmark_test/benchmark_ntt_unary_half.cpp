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

    constexpr size_t N = NTT_VLEN / (sizeof(_Float16) * 8);
    benchmark_ntt_unary_half<ntt::ops::abs, _Float16, N>("abs", -(_Float16)10.f,
                                                         (_Float16)10.f);
    benchmark_ntt_unary_half<ntt::ops::acos, _Float16, N>(
        "acos", -(_Float16)1.f, (_Float16)1.f);
    benchmark_ntt_unary_half<ntt::ops::acosh, _Float16, N>(
        "acosh", (_Float16)1.f, (_Float16)10.f);
    benchmark_ntt_unary_half<ntt::ops::asin, _Float16, N>(
        "asin", -(_Float16)1.f, (_Float16)1.f);
    benchmark_ntt_unary_half<ntt::ops::asinh, _Float16, N>(
        "asinh", -(_Float16)10.f, (_Float16)10.f);
    benchmark_ntt_unary_half<ntt::ops::ceil, _Float16, N>(
        "ceil", -(_Float16)10.f, (_Float16)10.f);
    benchmark_ntt_unary_half<ntt::ops::copy, _Float16, N>(
        "copy", -(_Float16)10.f, (_Float16)10.f);
    benchmark_ntt_unary_half<ntt::ops::cos, _Float16, N>("cos", -(_Float16)10.f,
                                                         (_Float16)10.f);
    benchmark_ntt_unary_half<ntt::ops::cosh, _Float16, N>(
        "cosh", -(_Float16)10.f, (_Float16)10.f);
    benchmark_ntt_unary_half<ntt::ops::erf, _Float16, N>("erf", -(_Float16)10.f,
                                                         (_Float16)10.f);
    benchmark_ntt_unary_half<ntt::ops::exp, _Float16, N>("exp", -(_Float16)10.f,
                                                         (_Float16)10.f);
    benchmark_ntt_unary_half<ntt::ops::floor, _Float16, N>(
        "floor", -(_Float16)10.f, (_Float16)10.f);
    benchmark_ntt_unary_half<ntt::ops::log, _Float16, N>("log", -(_Float16)10.f,
                                                         (_Float16)10.f);
    benchmark_ntt_unary_half<ntt::ops::neg, _Float16, N>("neg", -(_Float16)10.f,
                                                         (_Float16)10.f);
    benchmark_ntt_unary_half<ntt::ops::round, _Float16, N>(
        "round", -(_Float16)10.f, (_Float16)10.f);
    benchmark_ntt_unary_half<ntt::ops::rsqrt, _Float16, N>(
        "rsqrt", (_Float16)1.f, (_Float16)10.f);
    benchmark_ntt_unary_half<ntt::ops::sign, _Float16, N>(
        "sign", -(_Float16)10.f, (_Float16)10.f);
    benchmark_ntt_unary_half<ntt::ops::sin, _Float16, N>("sin", -(_Float16)10.f,
                                                         (_Float16)10.f);
    benchmark_ntt_unary_half<ntt::ops::sinh, _Float16, N>(
        "sinh", -(_Float16)10.f, (_Float16)10.f);
    benchmark_ntt_unary_half<ntt::ops::sqrt, _Float16, N>("sqrt", (_Float16)1.f,
                                                          (_Float16)10.f);
    benchmark_ntt_unary_half<ntt::ops::square, _Float16, N>(
        "square", -(_Float16)10.f, (_Float16)10.f);
    benchmark_ntt_unary_half<ntt::ops::swish, _Float16, N>(
        "swish", -(_Float16)10.f, (_Float16)10.f);
    benchmark_ntt_unary_half<ntt::ops::tanh, _Float16, N>(
        "tanh", -(_Float16)10.f, (_Float16)10.f);
}