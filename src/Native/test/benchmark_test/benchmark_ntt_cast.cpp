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
#include "../include/nncase/runtime/float8.h"
#include "ntt_test.h"
#include <iomanip>
#include <memory>
#include <nncase/ntt/ntt.h>
#include <string>

using namespace nncase;

template <typename T> struct TypeToString;

template <> struct TypeToString<float> {
    static constexpr char name[] = "float";
};

template <> struct TypeToString<int> {
    static constexpr char name[] = "int32";
};

template <> struct TypeToString<unsigned int> {
    static constexpr char name[] = "uint32";
};

template <> struct TypeToString<bool> {
    static constexpr char name[] = "bool";
};

template <> struct TypeToString<float_e4m3_t> {
    static constexpr char name[] = "f8e4m3";
};

template <typename T1, typename T2, size_t N>
void benchmark_ntt_cast(T1 init_low, T1 init_high) {
    std::string op = std::string(TypeToString<T1>::name) + "-" +
                     std::string(TypeToString<T2>::name);
    constexpr size_t warmup_size = 10;
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

    using tensor_type1 =
        ntt::tensor<ntt::vector<T1, N>, ntt::fixed_shape<size>>;
    using tensor_type2 =
        ntt::tensor<ntt::vector<T2, N>, ntt::fixed_shape<size>>;

    std::unique_ptr<tensor_type1> ntt_input(new tensor_type1);
    std::unique_ptr<tensor_type2> ntt_output(new tensor_type2);
    NttTest::init_tensor(*ntt_input, init_low, init_high);

    // warm up
    for (size_t i = 0; i < warmup_size; i++)
        ntt::cast(*ntt_input, *ntt_output);

    // run
    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_size; i++) {
        ntt::cast(*ntt_input, *ntt_output);
        asm volatile("" ::"g"(ntt_output));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << "_" << op << " took " << std::setprecision(1)
              << std::fixed << static_cast<float>(t2 - t1) / size / run_size
              << " cycles" << std::endl;
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    constexpr size_t N = NTT_VLEN / (sizeof(float) * 8);
    benchmark_ntt_cast<float, int, N>(-100.f, 100.f);
    benchmark_ntt_cast<int, float, N>(-100, 100);
    benchmark_ntt_cast<float, unsigned int, N>(0.f, 100.f);
    benchmark_ntt_cast<unsigned int, float, N>(0, 100);
    benchmark_ntt_cast<float, bool, N>(-100.f, 100.f);
    benchmark_ntt_cast<bool, float, N>(0, 1);
    benchmark_ntt_cast<float, float_e4m3_t, N>(-1000.f, 1000.f);
}
