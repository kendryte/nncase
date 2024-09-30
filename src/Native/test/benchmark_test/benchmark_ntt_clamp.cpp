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
#include <nncase/ntt/ntt.h>

using namespace nncase;

template <typename T, size_t N>
void benchmark_ntt_clamp(T init_low, T init_high, T clamp_low, T clamp_high) {
    std::string pack_mode = "NoPack";
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

    using tensor_type = ntt::tensor<ntt::vector<T, N>, ntt::fixed_shape<size>>;
    std::unique_ptr<tensor_type> ntt_input(new tensor_type);
    std::unique_ptr<tensor_type> ntt_output(new tensor_type);
    NttTest::init_tensor(*ntt_input, init_low, init_high);

    // warm up
    for (size_t i = 0; i < warmup_size; i++)
        ntt::clamp(*ntt_input, *ntt_output, clamp_low, clamp_high);

    // run
    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_size; i++) {
        ntt::clamp(*ntt_input, *ntt_output, clamp_low, clamp_high);
        asm volatile("" ::"g"(ntt_output));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << "_" << pack_mode << " took "
              << std::setprecision(1) << std::fixed
              << static_cast<float>(t2 - t1) / size / run_size << " cycles"
              << std::endl;
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    constexpr size_t N = NTT_VLEN / (sizeof(float) * 8);
    benchmark_ntt_clamp<float, N>(-10.f, 10.f, -6.f, 6.f);
}