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

#define BENCHMARMK_NTT_BINARY(op, dtype, N, lhs_low, lhs_high, rhs_low,        \
                              rhs_high)                                        \
    void benchmark_ntt_binary_##op##_##N() {                                   \
        constexpr size_t size = 2000;                                          \
        ntt::tensor<ntt::vector<dtype, N>, ntt::fixed_shape<size>> ntt_lhs;    \
        NttTest::init_tensor(ntt_lhs, lhs_low, lhs_high);                      \
                                                                               \
        ntt::tensor<ntt::vector<dtype, N>, ntt::fixed_shape<size>> ntt_rhs;    \
        NttTest::init_tensor(ntt_rhs, rhs_low, rhs_high);                      \
                                                                               \
        auto t1 = NttTest::get_cpu_cycle();                                    \
        for (size_t i = 0; i < size; i++)                                      \
            ntt::op(ntt_lhs, ntt_rhs);                                         \
        auto t2 = NttTest::get_cpu_cycle();                                    \
        std::cout << __FUNCTION__ << " took "                                  \
                  << static_cast<float>(t2 - t1) / size / size << " cycles"    \
                  << std::endl;                                                \
    }

#define REGISTER_NTT_BINARY(N)                                                 \
    BENCHMARMK_NTT_BINARY(add, float, N, -10.f, 10.f, -10.f, 10.f)             \
    BENCHMARMK_NTT_BINARY(sub, float, N, -10.f, 10.f, -10.f, 10.f)             \
    BENCHMARMK_NTT_BINARY(mul, float, N, -10.f, 10.f, -10.f, 10.f)             \
    BENCHMARMK_NTT_BINARY(div, float, N, -10.f, 10.f, 1.f, 10.f)               \
    BENCHMARMK_NTT_BINARY(max, float, N, -10.f, 10.f, -10.f, 10.f)             \
    BENCHMARMK_NTT_BINARY(min, float, N, -10.f, 10.f, -10.f, 10.f)             \
    BENCHMARMK_NTT_BINARY(floor_mod, int32_t, N, -10, 10, 1, 10)               \
    BENCHMARMK_NTT_BINARY(mod, float, N, -10.f, 10.f, 1.f, 10.f)               \
    BENCHMARMK_NTT_BINARY(pow, float, N, 0.f, 3.f, 0.f, 3.f)

REGISTER_NTT_BINARY(4)
REGISTER_NTT_BINARY(8)
REGISTER_NTT_BINARY(16)
REGISTER_NTT_BINARY(32)

#define RUN_NTT_BINARY(N)                                                      \
    benchmark_ntt_binary_add_##N();                                            \
    benchmark_ntt_binary_sub_##N();                                            \
    benchmark_ntt_binary_mul_##N();                                            \
    benchmark_ntt_binary_div_##N();                                            \
    benchmark_ntt_binary_max_##N();                                            \
    benchmark_ntt_binary_min_##N();                                            \
    benchmark_ntt_binary_floor_mod_##N();                                      \
    benchmark_ntt_binary_mod_##N();                                            \
    benchmark_ntt_binary_pow_##N();

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    RUN_NTT_BINARY(4)
    RUN_NTT_BINARY(8)
    RUN_NTT_BINARY(16)
    RUN_NTT_BINARY(32)
}