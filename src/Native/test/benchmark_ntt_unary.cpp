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
#include <nncase/ntt/ntt.h>

using namespace nncase;


#define BENCHMARMK_NTT_UNARY(op, dtype, low, high)                                                       \
void benchmark_ntt_unary_##op()                                                                          \
{                                                                                                        \
    constexpr size_t size = 10000;                                                                       \
    ntt::tensor<ntt::vector<dtype, 8>, ntt::fixed_shape<size>> ntt_input;                                \
    NttTest::init_tensor(ntt_input, low, high);                                                          \
                                                                                                         \
    auto t1 = NttTest::get_cycle();                                                                      \
    for (size_t i = 0; i < size; i++)                                                                    \
        ntt::op(ntt_input);                                                                              \
    auto t2 = NttTest::get_cycle();                                                                      \
    std::cout << __FUNCTION__ << " took " << static_cast<float>(t2 - t1) / size / size << " cycles"      \
                << std::endl;                                                                            \
}

BENCHMARMK_NTT_UNARY(abs, float, -10.f, 10.f)
BENCHMARMK_NTT_UNARY(acos, float, -1.f, 1.f)
BENCHMARMK_NTT_UNARY(acosh, float, 1.f, 10.f)
BENCHMARMK_NTT_UNARY(asin, float, -1.f, 1.f)
BENCHMARMK_NTT_UNARY(asinh, float, -10.f, 10.f)
BENCHMARMK_NTT_UNARY(ceil, float, -10.f, 10.f)
BENCHMARMK_NTT_UNARY(cos, float, -10.f, 10.f)
BENCHMARMK_NTT_UNARY(cosh, float, -10.f, 10.f)
BENCHMARMK_NTT_UNARY(exp, float, -10.f, 10.f)
BENCHMARMK_NTT_UNARY(floor, float, -10.f, 10.f)
BENCHMARMK_NTT_UNARY(log, float, -10.f, 10.f)
BENCHMARMK_NTT_UNARY(neg, float, -10.f, 10.f)
BENCHMARMK_NTT_UNARY(round, float, -10.f, 10.f)
BENCHMARMK_NTT_UNARY(rsqrt, float, 1.f, 10.f)
BENCHMARMK_NTT_UNARY(sign, float, -10.f, 10.f)
BENCHMARMK_NTT_UNARY(sin, float, -10.f, 10.f)
BENCHMARMK_NTT_UNARY(sinh, float, -10.f, 10.f)
BENCHMARMK_NTT_UNARY(sqrt, float, 1.f, 10.f)
BENCHMARMK_NTT_UNARY(square, float, -10.f, 10.f)
BENCHMARMK_NTT_UNARY(swish, float, -10.f, 10.f)
BENCHMARMK_NTT_UNARY(tanh, float, -10.f, 10.f)

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    benchmark_ntt_unary_abs();
    benchmark_ntt_unary_acos();
    benchmark_ntt_unary_acosh();
    benchmark_ntt_unary_asin();
    benchmark_ntt_unary_asinh();
    benchmark_ntt_unary_ceil();
    benchmark_ntt_unary_cos();
    benchmark_ntt_unary_cosh();
    benchmark_ntt_unary_exp();
    benchmark_ntt_unary_floor();
    benchmark_ntt_unary_log();
    benchmark_ntt_unary_neg();
    benchmark_ntt_unary_round();
    benchmark_ntt_unary_rsqrt();
    benchmark_ntt_unary_sign();
    benchmark_ntt_unary_sin();
    benchmark_ntt_unary_sinh();
    benchmark_ntt_unary_sqrt();
    benchmark_ntt_unary_square();
    benchmark_ntt_unary_swish();
    benchmark_ntt_unary_tanh();
}