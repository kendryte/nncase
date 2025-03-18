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
void benchmark_ntt_compare(std::string op_name, T lhs_low, T lhs_high, T rhs_low,
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

    using tensor_type1 = ntt::tensor<ntt::vector<bool, N>, ntt::fixed_shape<size2>>;
                
    tensor_type ntt_lhs, ntt_rhs;
    NttTest::init_tensor(ntt_lhs, lhs_low, lhs_high);
    NttTest::init_tensor(ntt_rhs, rhs_low, rhs_high);
    tensor_type1 ntt_result;

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < size1; i++) {
        ntt::compare<Op>(ntt_lhs, ntt_rhs, ntt_result);
        asm volatile("" ::"g"(ntt_result));
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

    constexpr size_t N = NTT_VLEN / (sizeof(float) * 8);
    benchmark_ntt_compare<ntt::ops::equal, float, N>("equal", -10.f, 10.f,
                                                     -10.f, 10.f);
    // benchmark_ntt_compare<ntt::ops::not_equal, float, N>("not_equal", -10.f,
    //                                                      10.f, -10.f, 10.f);
    // benchmark_ntt_compare<ntt::ops::greater, float, N>("greater", -10.f, 10.f,
    //                                                    -10.f, 10.f);
    // benchmark_ntt_compare<ntt::ops::greater_or_equal, float, N>(
    //     "greater_or_equal", -10.f, 10.f, 1.f, 10.f);
    // benchmark_ntt_compare<ntt::ops::less, float, N>("less", -10.f, 10.f, -10.f,
    //                                                 10.f);
    // benchmark_ntt_compare<ntt::ops::less_or_equal, float, N>(
    //     "less_or_equal", -10.f, 10.f, -10.f, 10.f);
}