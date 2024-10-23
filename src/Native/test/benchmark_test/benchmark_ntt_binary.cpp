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

template <template <typename T1, typename T2> class Op, typename T, size_t N>
void benchmark_ntt_binary(std::string op_name, T lhs_low, T lhs_high, T rhs_low,
                          T rhs_high) {
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
    using tensor_type = ntt::tensor<ntt::vector<T, N>, ntt::fixed_shape<size2>>;
    tensor_type ntt_lhs, ntt_rhs, ntt_result;
    NttTest::init_tensor(ntt_lhs, lhs_low, lhs_high);
    NttTest::init_tensor(ntt_rhs, rhs_low, rhs_high);
    Op<tensor_type, tensor_type> op;

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < size1; i++)
        ntt_result = op(ntt_lhs, ntt_rhs);
    auto t2 = NttTest::get_cpu_cycle();
#if __x86_64__
    asm volatile("" ::"g"(ntt_result));
#endif
    std::cout << __FUNCTION__ << "_" << op_name << " took "
              << std::setprecision(1) << std::fixed
              << static_cast<float>(t2 - t1) / size1 / size2 << " cycles"
              << std::endl;
}

#define BENCHMARK_NTT_BINARY(OP)                                               \
    template <size_t N, size_t run_size, size_t size>                          \
    void benchmark_ntt_unary_##OP() {                                          \
                                                                               \
        using tensor_type1 =                                                   \
            ntt::tensor<ntt::vector<float, N>, ntt::fixed_shape<size>>;        \
        using tensor_type2 =                                                   \
            ntt::tensor<ntt::vector<float, N>, ntt::fixed_shape<size>>;        \
        using tensor_type_out =                                                \
            ntt::tensor<ntt::vector<float, N>, ntt::fixed_shape<size>>;        \
        constexpr size_t warmup_size = 30;                                     \
                                                                               \
        tensor_type1 ntt_input1;                                               \
        tensor_type2 ntt_input2;                                               \
        tensor_type_out ntt_output;                                            \
        NttTest::init_tensor(ntt_input1, -10.f, 10.f);                         \
        NttTest::init_tensor(ntt_input2, -10.f, 10.f);                         \
                                                                               \
        for (size_t i = 0; i < warmup_size; i++)                               \
            ntt::OP(ntt_input1, ntt_input2, ntt_output);                       \
                                                                               \
        auto t1 = NttTest::get_cpu_cycle();                                    \
        for (size_t i = 0; i < run_size; i++) {                                \
            ntt::OP(ntt_input1, ntt_input2, ntt_output);                       \
            asm volatile("" ::"g"(ntt_output));                                \
            asm volatile("" ::"g"(ntt_input1));                                \
            asm volatile("" ::"g"(ntt_input2));                                \
        }                                                                      \
        auto t2 = NttTest::get_cpu_cycle();                                    \
                                                                               \
        std::cout << __FUNCTION__ << " took " << std::setprecision(1)          \
                  << std::fixed                                                \
                  << static_cast<float>(t2 - t1) / size / run_size             \
                  << " cycles" << std::endl;                                   \
    }

BENCHMARK_NTT_BINARY(add)
BENCHMARK_NTT_BINARY(div)
BENCHMARK_NTT_BINARY(max)
BENCHMARK_NTT_BINARY(min)
BENCHMARK_NTT_BINARY(mod)
BENCHMARK_NTT_BINARY(mul)
BENCHMARK_NTT_BINARY(sub)

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
    benchmark_ntt_binary<ntt::ops::floor_mod, int32_t, N>("floor_mod", -10, 10,
                                                          1, 10);
    benchmark_ntt_binary<ntt::ops::pow, float, N>("pow", 0.f, 3.f, 0.f, 3.f);

    benchmark_ntt_unary_add<N, run_size, size>();
    benchmark_ntt_unary_div<N, run_size, size>();
    benchmark_ntt_unary_max<N, run_size, size>();
    benchmark_ntt_unary_min<N, run_size, size>();
    benchmark_ntt_unary_mod<N, run_size, size>();
    benchmark_ntt_unary_mul<N, run_size, size>();
    benchmark_ntt_unary_sub<N, run_size, size>();
}