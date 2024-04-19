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

#define BENCHMARMK_NTT_UNARY(op, dtype, N, low, high)                          \
    void benchmark_ntt_unary_##op##_##N() {                                    \
        constexpr size_t size = 2000;                                          \
        ntt::tensor<ntt::vector<dtype, N>, ntt::fixed_shape<size>> ntt_input;  \
        NttTest::init_tensor(ntt_input, low, high);                            \
                                                                               \
        auto t1 = NttTest::get_cpu_cycle();                                    \
        for (size_t i = 0; i < size; i++)                                      \
            ntt::op(ntt_input);                                                \
        auto t2 = NttTest::get_cpu_cycle();                                    \
        std::cout << __FUNCTION__ << " took "                                  \
                  << static_cast<float>(t2 - t1) / size / size << " cycles"    \
                  << std::endl;                                                \
    }

#define REGISTER_NTT_UNARY(N)                                                  \
    BENCHMARMK_NTT_UNARY(abs, float, N, -10.f, 10.f)                           \
    BENCHMARMK_NTT_UNARY(acos, float, N, -1.f, 1.f)                            \
    BENCHMARMK_NTT_UNARY(acosh, float, N, 1.f, 10.f)                           \
    BENCHMARMK_NTT_UNARY(asin, float, N, -1.f, 1.f)                            \
    BENCHMARMK_NTT_UNARY(asinh, float, N, -10.f, 10.f)                         \
    BENCHMARMK_NTT_UNARY(ceil, float, N, -10.f, 10.f)                          \
    BENCHMARMK_NTT_UNARY(cos, float, N, -10.f, 10.f)                           \
    BENCHMARMK_NTT_UNARY(cosh, float, N, -10.f, 10.f)                          \
    BENCHMARMK_NTT_UNARY(exp, float, N, -10.f, 10.f)                           \
    BENCHMARMK_NTT_UNARY(floor, float, N, -10.f, 10.f)                         \
    BENCHMARMK_NTT_UNARY(log, float, N, -10.f, 10.f)                           \
    BENCHMARMK_NTT_UNARY(neg, float, N, -10.f, 10.f)                           \
    BENCHMARMK_NTT_UNARY(round, float, N, -10.f, 10.f)                         \
    BENCHMARMK_NTT_UNARY(rsqrt, float, N, 1.f, 10.f)                           \
    BENCHMARMK_NTT_UNARY(sign, float, N, -10.f, 10.f)                          \
    BENCHMARMK_NTT_UNARY(sin, float, N, -10.f, 10.f)                           \
    BENCHMARMK_NTT_UNARY(sinh, float, N, -10.f, 10.f)                          \
    BENCHMARMK_NTT_UNARY(sqrt, float, N, 1.f, 10.f)                            \
    BENCHMARMK_NTT_UNARY(square, float, N, -10.f, 10.f)                        \
    BENCHMARMK_NTT_UNARY(swish, float, N, -10.f, 10.f)                         \
    BENCHMARMK_NTT_UNARY(tanh, float, N, -10.f, 10.f)

REGISTER_NTT_UNARY(4)
REGISTER_NTT_UNARY(8)
REGISTER_NTT_UNARY(16)
REGISTER_NTT_UNARY(32)

#define RUN_NTT_UNARY(N)                                                       \
    benchmark_ntt_unary_abs_##N();                                             \
    benchmark_ntt_unary_acos_##N();                                            \
    benchmark_ntt_unary_acosh_##N();                                           \
    benchmark_ntt_unary_asin_##N();                                            \
    benchmark_ntt_unary_asinh_##N();                                           \
    benchmark_ntt_unary_ceil_##N();                                            \
    benchmark_ntt_unary_cos_##N();                                             \
    benchmark_ntt_unary_cosh_##N();                                            \
    benchmark_ntt_unary_exp_##N();                                             \
    benchmark_ntt_unary_floor_##N();                                           \
    benchmark_ntt_unary_log_##N();                                             \
    benchmark_ntt_unary_neg_##N();                                             \
    benchmark_ntt_unary_round_##N();                                           \
    benchmark_ntt_unary_rsqrt_##N();                                           \
    benchmark_ntt_unary_sign_##N();                                            \
    benchmark_ntt_unary_sin_##N();                                             \
    benchmark_ntt_unary_sinh_##N();                                            \
    benchmark_ntt_unary_sqrt_##N();                                            \
    benchmark_ntt_unary_square_##N();                                          \
    benchmark_ntt_unary_swish_##N();                                           \
    benchmark_ntt_unary_tanh_##N();

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    RUN_NTT_UNARY(4)
    RUN_NTT_UNARY(8)
    RUN_NTT_UNARY(16)
    RUN_NTT_UNARY(32)
}