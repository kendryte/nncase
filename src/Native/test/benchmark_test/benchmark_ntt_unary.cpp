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

#define BENCHMARMK_NTT_UNARY(op)                                               \
    template <typename T, size_t N>                                            \
    void benchmark_ntt_unary_##op(T low, T high) {                             \
        constexpr size_t size = 2000;                                          \
        ntt::tensor<ntt::vector<T, N>, ntt::fixed_shape<size>> ntt_input;      \
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

#define REGISTER_NTT_UNARY                                                     \
    BENCHMARMK_NTT_UNARY(abs)                                                  \
    BENCHMARMK_NTT_UNARY(acos)                                                 \
    BENCHMARMK_NTT_UNARY(acosh)                                                \
    BENCHMARMK_NTT_UNARY(asin)                                                 \
    BENCHMARMK_NTT_UNARY(asinh)                                                \
    BENCHMARMK_NTT_UNARY(ceil)                                                 \
    BENCHMARMK_NTT_UNARY(cos)                                                  \
    BENCHMARMK_NTT_UNARY(cosh)                                                 \
    BENCHMARMK_NTT_UNARY(exp)                                                  \
    BENCHMARMK_NTT_UNARY(floor)                                                \
    BENCHMARMK_NTT_UNARY(log)                                                  \
    BENCHMARMK_NTT_UNARY(neg)                                                  \
    BENCHMARMK_NTT_UNARY(round)                                                \
    BENCHMARMK_NTT_UNARY(rsqrt)                                                \
    BENCHMARMK_NTT_UNARY(sign)                                                 \
    BENCHMARMK_NTT_UNARY(sin)                                                  \
    BENCHMARMK_NTT_UNARY(sinh)                                                 \
    BENCHMARMK_NTT_UNARY(sqrt)                                                 \
    BENCHMARMK_NTT_UNARY(square)                                               \
    BENCHMARMK_NTT_UNARY(swish)                                                \
    BENCHMARMK_NTT_UNARY(tanh)

REGISTER_NTT_UNARY

#define RUN_NTT_UNARY(N)                                                       \
    benchmark_ntt_unary_abs<float, N>(-10.f, 10.f);                            \
    benchmark_ntt_unary_acos<float, N>(-1.f, 1.f);                             \
    benchmark_ntt_unary_acosh<float, N>(1.f, 10.f);                            \
    benchmark_ntt_unary_asin<float, N>(-1.f, 1.f);                             \
    benchmark_ntt_unary_asinh<float, N>(-10.f, 10.f);                          \
    benchmark_ntt_unary_ceil<float, N>(-10.f, 10.f);                           \
    benchmark_ntt_unary_cos<float, N>(-10.f, 10.f);                            \
    benchmark_ntt_unary_cosh<float, N>(-10.f, 10.f);                           \
    benchmark_ntt_unary_exp<float, N>(-10.f, 10.f);                            \
    benchmark_ntt_unary_floor<float, N>(-10.f, 10.f);                          \
    benchmark_ntt_unary_log<float, N>(-10.f, 10.f);                            \
    benchmark_ntt_unary_neg<float, N>(-10.f, 10.f);                            \
    benchmark_ntt_unary_round<float, N>(-10.f, 10.f);                          \
    benchmark_ntt_unary_rsqrt<float, N>(1.f, 10.f);                            \
    benchmark_ntt_unary_sign<float, N>(-10.f, 10.f);                           \
    benchmark_ntt_unary_sin<float, N>(-10.f, 10.f);                            \
    benchmark_ntt_unary_sinh<float, N>(-10.f, 10.f);                           \
    benchmark_ntt_unary_sqrt<float, N>(1.f, 10.f);                             \
    benchmark_ntt_unary_square<float, N>(-10.f, 10.f);                         \
    benchmark_ntt_unary_swish<float, N>(-10.f, 10.f);                          \
    benchmark_ntt_unary_tanh<float, N>(-10.f, 10.f);

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    RUN_NTT_UNARY(NTT_VLEN / (sizeof(float) * 8))
}