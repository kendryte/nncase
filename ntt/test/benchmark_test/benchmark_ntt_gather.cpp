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
#include <cstddef>
#include <iomanip>
#include <memory>
#include <nncase/ntt/ntt.h>

using namespace nncase;

void benchmark_ntt_gather_pack1d_dim0_contiguous() {
    constexpr size_t warmup_size = 10;
#if __riscv
    constexpr size_t run_size = 300;
#elif __x86_64__
    constexpr size_t run_size = 2000;
#else
    constexpr size_t run_size = 2000;
#endif
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 32;
    constexpr size_t N = 128;
    constexpr size_t Period = 1;
    using tensor_a_type = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    using tensor_b_type = ntt::tensor<size_t, ntt::fixed_shape<1, M / Period>>;
    using tensor_pa_type =
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M, N / P>>;
    using tensor_pc_type = ntt::tensor<ntt::vector<float, P>,
                                       ntt::fixed_shape<1, M / Period, N / P>>;

    tensor_a_type ta;
    tensor_b_type tb;
    tensor_pa_type pa;
    tensor_pc_type pc;
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
    std::iota(tb.elements().begin(), tb.elements().end(), 0);
    std::ranges::for_each(tb.elements(), [](size_t &x) { x *= Period; });
    ntt::pack<1>(ta, pa);

    // warm up
    for (size_t i = 0; i < warmup_size; i++) {
        ntt::gather<0>(pa, tb, pc);
#if __x86_64__
        asm volatile("" ::"g"(pc));
#endif
    }

    // run
    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_size; i++) {
        ntt::gather<0>(pa, tb, pc);
#if __x86_64__
        asm volatile("" ::"g"(pc));
#endif
    }
    auto t2 = NttTest::get_cpu_cycle();

    constexpr size_t size = pc.elements().size();
    std::cout << __FUNCTION__ << " took " << std::setprecision(1) << std::fixed
              << static_cast<float>(t2 - t1) / run_size / size << " cycles"
              << std::endl;
}

void benchmark_ntt_gather_pack1d_dim0_no_contiguous() {
    constexpr size_t warmup_size = 10;
#if __riscv
    constexpr size_t run_size = 300;
#elif __x86_64__
    constexpr size_t run_size = 2000;
#else
    constexpr size_t run_size = 2000;
#endif
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 32;
    constexpr size_t N = 128;
    constexpr size_t Period = 2;
    using tensor_a_type = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    using tensor_b_type = ntt::tensor<size_t, ntt::fixed_shape<1, M / Period>>;
    using tensor_pa_type =
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M, N / P>>;
    using tensor_pc_type = ntt::tensor<ntt::vector<float, P>,
                                       ntt::fixed_shape<1, M / Period, N / P>>;

    tensor_a_type ta;
    tensor_b_type tb;
    tensor_pa_type pa;
    tensor_pc_type pc;
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
    std::iota(tb.elements().begin(), tb.elements().end(), 0);
    std::ranges::for_each(tb.elements(), [](size_t &x) { x *= Period; });
    ntt::pack<1>(ta, pa);

    // warm up
    for (size_t i = 0; i < warmup_size; i++)
        ntt::gather<0>(pa, tb, pc);

    // run
    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_size; i++) {
        ntt::gather<0>(pa, tb, pc);
        asm volatile("" ::"g"(pc));
    }
    auto t2 = NttTest::get_cpu_cycle();

    constexpr size_t size = pc.elements().size();
    std::cout << __FUNCTION__ << " took " << std::setprecision(1) << std::fixed
              << static_cast<float>(t2 - t1) / run_size / size << " cycles"
              << std::endl;
}

void benchmark_ntt_gather_pack1d_dim1_contiguous() {
    constexpr size_t warmup_size = 10;
#if __riscv
    constexpr size_t run_size = 300;
#elif __x86_64__
    constexpr size_t run_size = 2000;
#else
    constexpr size_t run_size = 2000;
#endif
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 8;
    constexpr size_t N = 512;
    constexpr size_t Period = 1;
    using tensor_a_type = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    using tensor_b_type =
        ntt::tensor<size_t, ntt::fixed_shape<1, N / P / Period>>;
    using tensor_pa_type =
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M, N / P>>;
    using tensor_pc_type = ntt::tensor<ntt::vector<float, P>,
                                       ntt::fixed_shape<M, 1, N / P / Period>>;

    tensor_a_type ta;
    tensor_b_type tb;
    tensor_pa_type pa;
    tensor_pc_type pc;
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
    std::iota(tb.elements().begin(), tb.elements().end(), 0);
    std::ranges::for_each(tb.elements(), [](size_t &x) { x *= Period; });
    ntt::pack<1>(ta, pa);

    // warm up
    for (size_t i = 0; i < warmup_size; i++)
        ntt::gather<1>(pa, tb, pc);

    // run
    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_size; i++) {
        ntt::gather<1>(pa, tb, pc);
        asm volatile("" ::"g"(pc));
    }
    auto t2 = NttTest::get_cpu_cycle();

    constexpr size_t size = pc.elements().size();
    std::cout << __FUNCTION__ << " took " << std::setprecision(1) << std::fixed
              << static_cast<float>(t2 - t1) / run_size / size << " cycles"
              << std::endl;
}

void benchmark_ntt_gather_pack1d_dim1_no_contiguous() {
    constexpr size_t warmup_size = 10;
#if __riscv
    constexpr size_t run_size = 300;
#elif __x86_64__
    constexpr size_t run_size = 2000;
#else
    constexpr size_t run_size = 2000;
#endif
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 64;
    constexpr size_t N = 64;
    constexpr size_t Period = 2;
    using tensor_a_type = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    using tensor_b_type =
        ntt::tensor<size_t, ntt::fixed_shape<1, N / P / Period>>;
    using tensor_pa_type =
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M, N / P>>;
    using tensor_pc_type = ntt::tensor<ntt::vector<float, P>,
                                       ntt::fixed_shape<M, 1, N / P / Period>>;

    tensor_a_type ta;
    tensor_b_type tb;
    tensor_pa_type pa;
    tensor_pc_type pc;
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
    std::iota(tb.elements().begin(), tb.elements().end(), 0);
    std::ranges::for_each(tb.elements(), [](size_t &x) { x *= Period; });
    ntt::pack<1>(ta, pa);

    // warm up
    for (size_t i = 0; i < warmup_size; i++)
        ntt::gather<1>(pa, tb, pc);

    // run
    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_size; i++) {
        ntt::gather<1>(pa, tb, pc);
        asm volatile("" ::"g"(pc));
    }
    auto t2 = NttTest::get_cpu_cycle();

    constexpr size_t size = pc.elements().size();
    std::cout << __FUNCTION__ << " took " << std::setprecision(1) << std::fixed
              << static_cast<float>(t2 - t1) / run_size / size << " cycles"
              << std::endl;
}

void benchmark_ntt_gather_pack2d_dim0_contiguous() {
    constexpr size_t warmup_size = 10;
#if __riscv
    constexpr size_t run_size = 300;
#elif __x86_64__
    constexpr size_t run_size = 2000;
#else
    constexpr size_t run_size = 2000;
#endif
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 64;
    constexpr size_t N = 64;
    using tensor_a_type = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    using tensor_b_type = ntt::tensor<size_t, ntt::fixed_shape<1, M / P>>;
    using tensor_pa_type =
        ntt::tensor<ntt::vector<float, P, P>, ntt::fixed_shape<M / P, N / P>>;
    using tensor_pc_type = ntt::tensor<ntt::vector<float, P, P>,
                                       ntt::fixed_shape<1, M / P, N / P>>;

    tensor_a_type ta;
    tensor_b_type tb;
    tensor_pa_type pa;
    tensor_pc_type pc;
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
    std::iota(tb.elements().begin(), tb.elements().end(), 0);
    NttTest::init_tensor(pa, -10.f, 10.f);
    ntt::pack<0, 1>(ta, pa);

    // warm up
    for (size_t i = 0; i < warmup_size; i++)
        ntt::gather<0>(pa, tb, pc);

    // run
    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_size; i++) {
        ntt::gather<0>(pa, tb, pc);
        asm volatile("" ::"g"(pc));
    }
    auto t2 = NttTest::get_cpu_cycle();

    constexpr size_t size = pc.elements().size() * P;
    std::cout << __FUNCTION__ << " took " << std::setprecision(1) << std::fixed
              << static_cast<float>(t2 - t1) / run_size / size << " cycles"
              << std::endl;
}

void benchmark_ntt_gather_pack2d_dim1_contiguous() {
    constexpr size_t warmup_size = 10;
#if __riscv
    constexpr size_t run_size = 300;
#elif __x86_64__
    constexpr size_t run_size = 2000;
#else
    constexpr size_t run_size = 2000;
#endif
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 64;
    constexpr size_t N = 64;
    using tensor_a_type = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    using tensor_b_type = ntt::tensor<size_t, ntt::fixed_shape<1, N / P>>;
    using tensor_pa_type =
        ntt::tensor<ntt::vector<float, P, P>, ntt::fixed_shape<M / P, N / P>>;
    using tensor_pc_type = ntt::tensor<ntt::vector<float, P, P>,
                                       ntt::fixed_shape<M / P, 1, N / P>>;

    tensor_a_type ta;
    tensor_b_type tb;
    tensor_pa_type pa;
    tensor_pc_type pc;
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
    std::iota(tb.elements().begin(), tb.elements().end(), 0);
    ntt::pack<0, 1>(ta, pa);

    // warm up
    for (size_t i = 0; i < warmup_size; i++)
        ntt::gather<1>(pa, tb, pc);

    // run
    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_size; i++) {
        ntt::gather<1>(pa, tb, pc);
        asm volatile("" ::"g"(pc));
    }
    auto t2 = NttTest::get_cpu_cycle();

    constexpr size_t size = pc.elements().size() * P;
    std::cout << __FUNCTION__ << " took " << std::setprecision(1) << std::fixed
              << static_cast<float>(t2 - t1) / run_size / size << " cycles"
              << std::endl;
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    benchmark_ntt_gather_pack1d_dim0_contiguous();
    benchmark_ntt_gather_pack1d_dim0_no_contiguous();
    benchmark_ntt_gather_pack1d_dim1_contiguous();
    benchmark_ntt_gather_pack1d_dim1_no_contiguous();
    benchmark_ntt_gather_pack2d_dim0_contiguous();
    benchmark_ntt_gather_pack2d_dim1_contiguous();
}