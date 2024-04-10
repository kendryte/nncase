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

#define BENCHMARMK_NTT_BINARY(op, dtype, lhs_low, lhs_high, rhs_low, rhs_high) \
    void benchmark_ntt_binary_##op() {                                         \
        constexpr size_t size = 10000;                                         \
        ntt::tensor<ntt::vector<dtype, 8>, ntt::fixed_shape<size>> ntt_lhs;    \
        NttTest::init_tensor(ntt_lhs, lhs_low, lhs_high);                      \
                                                                               \
        ntt::tensor<ntt::vector<dtype, 8>, ntt::fixed_shape<size>> ntt_rhs;    \
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

BENCHMARMK_NTT_BINARY(add, float, -10.f, 10.f, -10.f, 10.f)
BENCHMARMK_NTT_BINARY(sub, float, -10.f, 10.f, -10.f, 10.f)
BENCHMARMK_NTT_BINARY(mul, float, -10.f, 10.f, -10.f, 10.f)
BENCHMARMK_NTT_BINARY(div, float, -10.f, 10.f, 1.f, 10.f)
BENCHMARMK_NTT_BINARY(max, float, -10.f, 10.f, -10.f, 10.f)
BENCHMARMK_NTT_BINARY(min, float, -10.f, 10.f, -10.f, 10.f)
BENCHMARMK_NTT_BINARY(floor_mod, int32_t, -10, 10, 1, 10)
BENCHMARMK_NTT_BINARY(mod, float, -10.f, 10.f, 1.f, 10.f)
BENCHMARMK_NTT_BINARY(pow, float, 0.f, 3.f, 0.f, 3.f)

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    benchmark_ntt_binary_add();
    benchmark_ntt_binary_sub();
    benchmark_ntt_binary_mul();
    benchmark_ntt_binary_div();
    benchmark_ntt_binary_max();
    benchmark_ntt_binary_min();
    benchmark_ntt_binary_floor_mod();
    benchmark_ntt_binary_mod();
    benchmark_ntt_binary_pow();
}