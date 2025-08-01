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
#include <string>

using namespace nncase;

template <typename T, size_t N, size_t in_dim0, size_t in_dim1,
          size_t start_dim0, size_t start_dim1, size_t stop_dim0,
          size_t stop_dim1, size_t step_dim0, size_t step_dim1>
void benchmark_ntt_slice(T init_low, T init_high, const std::string &mode) {
    constexpr size_t warmup_size = 10;
    constexpr size_t axes[] = {0, 1};

#if __riscv
    constexpr size_t run_size = 300;
#elif __x86_64__
    constexpr size_t run_size = 2000;
#else
    constexpr size_t run_size = 300;
#endif

    constexpr size_t out_dim0 = (stop_dim0 - 1 - start_dim0) / step_dim0 + 1;
    constexpr size_t out_dim1 = (stop_dim1 - 1 - start_dim1) / step_dim1 + 1;

    // init
    auto ntt_input = ntt::make_tensor<ntt::vector<T, N>>(
        ntt::fixed_shape_v<in_dim0, in_dim1>);
    NttTest::init_tensor(ntt_input, init_low, init_high);
    auto ntt_output = ntt::make_tensor<ntt::vector<T, N>>(
        ntt::fixed_shape_v<out_dim0, out_dim1>);

    // warm up
    for (size_t i = 0; i < warmup_size; i++) {
        ntt::slice(ntt_input, ntt_output,
                   ntt::fixed_shape_v<start_dim0, start_dim1>,
                   ntt::fixed_shape_v<stop_dim0, stop_dim1>,
                   ntt::fixed_shape_v<axes[0], axes[1]>,
                   ntt::fixed_shape_v<step_dim0, step_dim1>);
        asm volatile("" ::"g"(ntt_output));
    }

    // run
    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_size; i++) {
        ntt::slice(ntt_input, ntt_output,
                   ntt::fixed_shape_v<start_dim0, start_dim1>,
                   ntt::fixed_shape_v<stop_dim0, stop_dim1>,
                   ntt::fixed_shape_v<axes[0], axes[1]>,
                   ntt::fixed_shape_v<step_dim0, step_dim1>);
        asm volatile("" ::"g"(ntt_output));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << "_" << mode << " took " << std::setprecision(1)
              << std::fixed
              << static_cast<float>(t2 - t1) / out_dim0 / out_dim1 / run_size
              << " cycles" << std::endl;
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    constexpr size_t N = NTT_VLEN / (sizeof(float) * 8);
    {
        constexpr size_t in_dim0 = 12;
        constexpr size_t in_dim1 = 64;
        benchmark_ntt_slice<float, N, in_dim0, in_dim1, 0, 0, in_dim0 / 2,
                            in_dim1, 1, 1>(-100.f, 100.f, "contiguous_step_1");
    }
    {
        constexpr size_t in_dim0 = 12;
        constexpr size_t in_dim1 = 64;
        benchmark_ntt_slice<float, N, in_dim0, in_dim1, 0, 0, in_dim0 / 2,
                            in_dim1, 2, 2>(-100.f, 100.f, "contiguous_step_2");
    }
    {
        constexpr size_t in_dim0 = 12;
        constexpr size_t in_dim1 = 64;
        benchmark_ntt_slice<float, N, in_dim0, in_dim1, 0, 0, in_dim0 / 2,
                            in_dim1, 4, 4>(-100.f, 100.f, "contiguous_step_4");
    }
    {
#if __riscv
        constexpr size_t in_dim0 = 12;
        constexpr size_t in_dim1 = 64;
#elif __x86_64__
        constexpr size_t in_dim0 = 64;
        constexpr size_t in_dim1 = 128;
#else
        constexpr size_t in_dim0 = 12;
        constexpr size_t in_dim1 = 64;
#endif
        benchmark_ntt_slice<float, N, in_dim0, in_dim1, 0, 0, in_dim0 / 2,
                            in_dim1 / 2, 1, 1>(-100.f, 100.f,
                                               "no_contiguous_step_1");
    }
    {
#if __riscv
        constexpr size_t in_dim0 = 12;
        constexpr size_t in_dim1 = 64;
#elif __x86_64__
        constexpr size_t in_dim0 = 64;
        constexpr size_t in_dim1 = 64;
#else
        constexpr size_t in_dim0 = 12;
        constexpr size_t in_dim1 = 64;
#endif
        benchmark_ntt_slice<float, N, in_dim0, in_dim1, 0, 0, in_dim0 / 2,
                            in_dim1 / 2, 2, 2>(-100.f, 100.f,
                                               "no_contiguous_step_2");
    }
    {
#if __riscv
        constexpr size_t in_dim0 = 12;
        constexpr size_t in_dim1 = 64;
#elif __x86_64__
        constexpr size_t in_dim0 = 128;
        constexpr size_t in_dim1 = 128;
#else
        constexpr size_t in_dim0 = 12;
        constexpr size_t in_dim1 = 64;
#endif
        benchmark_ntt_slice<float, N, in_dim0, in_dim1, 0, 0, in_dim0 / 2,
                            in_dim1 / 2, 4, 4>(-100.f, 100.f,
                                               "no_contiguous_step_4");
    }
}