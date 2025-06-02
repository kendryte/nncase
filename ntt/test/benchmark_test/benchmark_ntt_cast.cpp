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
#include <memory>
#include <nncase/bfloat16.h>
#include <nncase/float8.h>
#include <nncase/half.h>
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

template <> struct TypeToString<bfloat16> {
    static constexpr char name[] = "bfloat16";
};

template <> struct TypeToString<half> {
    static constexpr char name[] = "half";
};

template <typename T1, typename T2>
void benchmark_ntt_cast(T1 init_low, T1 init_high) {
    std::string op = std::string(TypeToString<T1>::name) + "-" +
                     std::string(TypeToString<T2>::name);
    constexpr size_t warmup_size = 10;
#if __riscv
    constexpr size_t run_size = 300;
    constexpr size_t size = 3200;
#elif __x86_64__
    constexpr size_t run_size = 2000;
    constexpr size_t size = 16000;
#else
    constexpr size_t run_size = 2000;
    constexpr size_t size = 3200;
#endif

    constexpr size_t M = NTT_VLEN / (sizeof(T1) * 8);
    constexpr size_t N = NTT_VLEN / (sizeof(T2) * 8);
    constexpr auto ratio = sizeof(T1) / sizeof(T2);
    constexpr size_t times = ratio <= 1 ? 1 : ratio;

    using tensor_type1 =
        ntt::tensor<ntt::vector<T1, M>, ntt::fixed_shape<size / M>>;
    using tensor_type2 =
        ntt::tensor<ntt::vector<T2, N>, ntt::fixed_shape<size / N>>;

    std::unique_ptr<tensor_type1> ntt_input(new tensor_type1);
    std::unique_ptr<tensor_type2> ntt_output(new tensor_type2);
    NttTest::init_tensor(*ntt_input, init_low, init_high);

    // warm up
    for (size_t i = 0; i < warmup_size; i++) {
        ntt::cast(*ntt_input, *ntt_output);
        asm volatile("" ::"g"(ntt_output));
    }

    // run
    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_size; i++) {
        ntt::cast(*ntt_input, *ntt_output);
        asm volatile("" ::"g"(ntt_output));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << "_" << op << " took " << std::setprecision(1)
              << std::fixed
              << static_cast<float>(t2 - t1) / (size / M) * times / run_size
              << " cycles" << std::endl;
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    benchmark_ntt_cast<float, int>(-100.f, 100.f);
    benchmark_ntt_cast<int, float>(-100, 100);
    benchmark_ntt_cast<float, unsigned int>(0.f, 100.f);
    benchmark_ntt_cast<unsigned int, float>(0, 100);
    benchmark_ntt_cast<float, bool>(-100.f, 100.f);
    benchmark_ntt_cast<bool, float>(0, 1);
    benchmark_ntt_cast<float, float_e4m3_t>(-1000.f, 1000.f);
    // benchmark_ntt_cast<float_e4m3_t, float>((float_e4m3_t)-448.f,
    //                                         (float_e4m3_t)448.f);
    // benchmark_ntt_cast<float_e4m3_t, bfloat16>((float_e4m3_t)-448.f,
    //                                            (float_e4m3_t)448.f);
    // benchmark_ntt_cast<float_e4m3_t, half>((float_e4m3_t)-448.f,
    //                                        (float_e4m3_t)448.f);
}
