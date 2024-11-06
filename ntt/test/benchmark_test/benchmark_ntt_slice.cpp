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

template <typename T, size_t N>
void benchmark_ntt_slice(T init_low, T init_high) {
    constexpr size_t warmup_size = 10;
    std::string pack_mode = "Pack";
#if __riscv
    constexpr size_t run_size = 300;
    constexpr size_t size1 = 3;
    constexpr size_t size2 = 256;
    constexpr size_t starts[] = {0, 0};
    constexpr size_t stops[] = {size1, size2};
    constexpr size_t axes[] = {0, 1};
    constexpr size_t steps[] = {1, 1};
#elif __x86_64__
    constexpr size_t run_size = 2000;
    constexpr size_t size1 = 3;
    constexpr size_t size2 = 256;
    constexpr size_t starts[] = {0, 0};
    constexpr size_t stops[] = {size1, size2};
    constexpr size_t axes[] = {0, 1};
    constexpr size_t steps[] = {1, 1};
#else
    constexpr size_t run_size = 300;
    constexpr size_t size1 = 3;
    constexpr size_t size2 = 256;
    constexpr size_t starts[] = {0, 0};
    constexpr size_t stops[] = {size1, size2};
    constexpr size_t axes[] = {0, 1};
    constexpr size_t steps[] = {1, 1};
#endif
    using tensor_type1 =
        ntt::tensor<ntt::vector<T, N>, ntt::fixed_shape<1024, 256>>;
    using tensor_type2 =
        ntt::tensor<ntt::vector<T, N>, ntt::fixed_shape<size1, size2>>;

    // init
    std::unique_ptr<tensor_type1> ntt_input(new tensor_type1);
    NttTest::init_tensor(*ntt_input, init_low, init_high);
    std::unique_ptr<tensor_type2> ntt_output(new tensor_type2);

    // warm up
    for (size_t i = 0; i < warmup_size; i++)
        ntt::slice<ntt::fixed_shape<starts[0], starts[1]>,
                   ntt::fixed_shape<stops[0], stops[1]>,
                   ntt::fixed_shape<axes[0], axes[1]>,
                   ntt::fixed_shape<steps[0], steps[1]>>(*ntt_input,
                                                         *ntt_output);

    // run
    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_size; i++) {
        ntt::slice<ntt::fixed_shape<starts[0], starts[1]>,
                   ntt::fixed_shape<stops[0], stops[1]>,
                   ntt::fixed_shape<axes[0], axes[1]>,
                   ntt::fixed_shape<steps[0], steps[1]>>(*ntt_input,
                                                         *ntt_output);
        asm volatile("" ::"g"(ntt_output));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << "_" << pack_mode << " took "
              << std::setprecision(1) << std::fixed
              << static_cast<float>(t2 - t1) / size1 / size2 / run_size
              << " cycles" << std::endl;
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    constexpr size_t N = NTT_VLEN / (sizeof(float) * 8);
    benchmark_ntt_slice<float, N>(-100.f, 100.f);
}