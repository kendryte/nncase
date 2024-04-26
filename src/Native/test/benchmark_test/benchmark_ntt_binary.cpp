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

#define BENCHMARMK_NTT_BINARY(op)                                              \
    template <typename T, size_t N>                                            \
    void benchmark_ntt_binary_##op(T lhs_low, T lhs_high, T rhs_low,           \
                                   T rhs_high) {                               \
        constexpr size_t size = 2000;                                          \
        ntt::tensor<ntt::vector<T, N>, ntt::fixed_shape<size>> ntt_lhs,        \
            ntt_rhs;                                                           \
        NttTest::init_tensor(ntt_lhs, lhs_low, lhs_high);                      \
        NttTest::init_tensor(ntt_rhs, rhs_low, rhs_high);                      \
                                                                               \
        auto t1 = NttTest::get_cpu_cycle();                                    \
        for (size_t i = 0; i < size; i++)                                      \
            ntt::op(ntt_lhs, ntt_rhs);                                         \
        auto t2 = NttTest::get_cpu_cycle();                                    \
        std::cout << __FUNCTION__ << " took " << std::setprecision(1)          \
                  << std::fixed << static_cast<float>(t2 - t1) / size / size   \
                  << " cycles" << std::endl;                                   \
    }

#define REGISTER_NTT_BINARY                                                    \
    BENCHMARMK_NTT_BINARY(add)                                                 \
    BENCHMARMK_NTT_BINARY(sub)                                                 \
    BENCHMARMK_NTT_BINARY(mul)                                                 \
    BENCHMARMK_NTT_BINARY(div)                                                 \
    BENCHMARMK_NTT_BINARY(max)                                                 \
    BENCHMARMK_NTT_BINARY(min)                                                 \
    BENCHMARMK_NTT_BINARY(floor_mod)                                           \
    BENCHMARMK_NTT_BINARY(mod)                                                 \
    BENCHMARMK_NTT_BINARY(pow)

REGISTER_NTT_BINARY

#define RUN_NTT_BINARY(N)                                                      \
    benchmark_ntt_binary_add<float, N>(-10.f, 10.f, -10.f, 10.f);              \
    benchmark_ntt_binary_sub<float, N>(-10.f, 10.f, -10.f, 10.f);              \
    benchmark_ntt_binary_mul<float, N>(-10.f, 10.f, -10.f, 10.f);              \
    benchmark_ntt_binary_div<float, N>(-10.f, 10.f, 1.f, 10.f);                \
    benchmark_ntt_binary_max<float, N>(-10.f, 10.f, -10.f, 10.f);              \
    benchmark_ntt_binary_min<float, N>(-10.f, 10.f, -10.f, 10.f);              \
    benchmark_ntt_binary_floor_mod<int32_t, N>(-10, 10, 1, 10);                \
    benchmark_ntt_binary_mod<float, N>(-10.f, 10.f, 1.f, 10.f);                \
    benchmark_ntt_binary_pow<float, N>(0.f, 3.f, 0.f, 3.f);

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    RUN_NTT_BINARY(NTT_VLEN / (sizeof(float) * 8))
}