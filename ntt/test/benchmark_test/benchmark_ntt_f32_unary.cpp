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
    constexpr size_t size1 = 1;
    constexpr size_t size2 = 129;
#elif __x86_64__
    constexpr size_t size1 = 1000;
    constexpr size_t size2 = 1000;
#else
    constexpr size_t size1 = 2000;
    constexpr size_t size2 = 2000;
#endif
    auto ntt_input =
        ntt::make_tensor<ntt::vector<T, N>>(ntt::fixed_shape_v<size2>);
    auto ntt_output =
        ntt::make_tensor<ntt::vector<T, N>>(ntt::fixed_shape_v<size2>);
    NttTest::init_tensor(ntt_input, low, high);

    // ntt::apply(ntt_input.shape(), [&](auto index) {
    //     for (size_t i = 0; i < N; i++) {
    //         ntt_input(index)(i) = 0;
    //     }
    // });

    for (size_t i = 0; i < size1; i++) {
        ntt::unary<Op>(ntt_input, ntt_output);
        asm volatile("" ::"g"(ntt_output));
    }

    auto t1 = NttTest::get_cpu_cycle();
    // for (size_t i = 0; i < size1; i++) {
    //     ntt::unary<Op>(ntt_input, ntt_output);
    //     asm volatile("" ::"g"(ntt_output));
    // }
    auto t2 = NttTest::get_cpu_cycle();

    auto cnt = 0;
    ntt::apply(ntt_input.shape(), [&](auto index) {
        for (size_t i = 0; i < N; i++) {
            std::cout << "ntt_input[" << cnt << "] = " << ntt_input(index)(i);
            std::cout << ", ntt_output[" << cnt
                      << "] = " << ntt_output(index)(i) << std::endl;
            cnt++;
        }
    });

    std::cout << __FUNCTION__ << "_" << op_name << " took "
              << std::setprecision(1) << std::fixed
              << static_cast<float>(t2 - t1) / size1 / size2 << " cycles"
              << std::endl;
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    constexpr size_t N = NTT_VLEN / (sizeof(float) * 8);
    // benchmark_ntt_unary<ntt::ops::copy, float, N>("copy", -10.f, 10.f);

    // benchmark_ntt_unary<ntt::ops::abs, float, N>("abs", -10.f, 10.f);
    // benchmark_ntt_unary<ntt::ops::acos, float, N>("acos", -1.f, 1.f);
    // benchmark_ntt_unary<ntt::ops::acosh, float, N>("acosh", 1.f, 10.f);
    // benchmark_ntt_unary<ntt::ops::asin, float, N>("asin", -1.f, 1.f);
    // benchmark_ntt_unary<ntt::ops::asinh, float, N>("asinh", -10.f, 10.f);
    // benchmark_ntt_unary<ntt::ops::ceil, float, N>("ceil", -10.f, 10.f);
    // benchmark_ntt_unary<ntt::ops::copy, float, N>("copy", -10.f, 10.f);
    // benchmark_ntt_unary<ntt::ops::cos, float, N>("cos", -10.f, 10.f);
    // benchmark_ntt_unary<ntt::ops::cosh, float, N>("cosh", -10.f, 10.f);
    // benchmark_ntt_unary<ntt::ops::erf, float, N>("erf", -10.f, 10.f);
    // benchmark_ntt_unary<ntt::ops::exp, float, N>("exp", -10.f, 10.f);
    // benchmark_ntt_unary<ntt::ops::floor, float, N>("floor", -10.f, 10.f);
    // benchmark_ntt_unary<ntt::ops::log, float, N>("log", -10.f, 10.f);
    // benchmark_ntt_unary<ntt::ops::neg, float, N>("neg", -10.f, 10.f);
    // benchmark_ntt_unary<ntt::ops::round, float, N>("round", -10.f, 10.f);
    // benchmark_ntt_unary<ntt::ops::rsqrt, float, N>("rsqrt", 1.f, 10.f);
    // benchmark_ntt_unary<ntt::ops::sign, float, N>("sign", -10.f, 10.f);
    // benchmark_ntt_unary<ntt::ops::sin, float, N>("sin", -10.f, 10.f);
    // benchmark_ntt_unary<ntt::ops::sinh, float, N>("sinh", -10.f, 10.f);
    benchmark_ntt_unary<ntt::ops::sqrt, float, N>("sqrt", 1.f, 10.f);
    // benchmark_ntt_unary<ntt::ops::square, float, N>("square", -10.f, 10.f);
    // benchmark_ntt_unary<ntt::ops::swish, float, N>("swish", -10.f, 10.f);
    // benchmark_ntt_unary<ntt::ops::tanh, float, N>("tanh", -10.f, 10.f);
}