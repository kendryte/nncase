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
    auto ta = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    auto tb = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<1, M / Period>);
    auto pa =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<M, N / P>);
    auto pc = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<1, M / Period, N / P>);
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
    std::iota(tb.elements().begin(), tb.elements().end(), 0);
    std::ranges::for_each(tb.elements(), [](int64_t &x) { x *= Period; });
    ntt::pack(ta, pa, ntt::fixed_shape_v<1>);

    // warm up
    for (size_t i = 0; i < warmup_size; i++) {
        ntt::gather(pa, tb, pc, 0_dim);
        asm volatile("" ::"g"(pc));
    }

    // run
    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_size; i++) {
        ntt::gather(pa, tb, pc, 0_dim);
        asm volatile("" ::"g"(pc));
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
    auto ta = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    auto tb = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<1, M / Period>);
    auto pa =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<M, N / P>);
    auto pc = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<1, M / Period, N / P>);
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
    std::iota(tb.elements().begin(), tb.elements().end(), 0);
    std::ranges::for_each(tb.elements(), [](int64_t &x) { x *= Period; });
    ntt::pack(ta, pa, ntt::fixed_shape_v<1>);

    // warm up
    for (size_t i = 0; i < warmup_size; i++) {
        ntt::gather(pa, tb, pc, 0_dim);
        asm volatile("" ::"g"(pc));
    }

    // run
    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_size; i++) {
        ntt::gather(pa, tb, pc, 0_dim);
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
    auto ta = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    auto tb = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<1, N / P / Period>);
    auto pa =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<M, N / P>);
    auto pc = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<M, 1, N / P / Period>);
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
    std::iota(tb.elements().begin(), tb.elements().end(), 0);
    std::ranges::for_each(tb.elements(), [](int64_t &x) { x *= Period; });
    ntt::pack(ta, pa, ntt::fixed_shape_v<1>);

    // warm up
    for (size_t i = 0; i < warmup_size; i++)
        ntt::gather(pa, tb, pc, 1_dim);

    // run
    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_size; i++) {
        ntt::gather(pa, tb, pc, 1_dim);
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
    auto ta = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    auto tb = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<1, N / P / Period>);
    auto pa =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<M, N / P>);
    auto pc = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<M, 1, N / P / Period>);
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
    std::iota(tb.elements().begin(), tb.elements().end(), 0);
    std::ranges::for_each(tb.elements(), [](int64_t &x) { x *= Period; });
    ntt::pack(ta, pa, ntt::fixed_shape_v<1>);

    // warm up
    for (size_t i = 0; i < warmup_size; i++)
        ntt::gather(pa, tb, pc, 1_dim);

    // run
    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_size; i++) {
        ntt::gather(pa, tb, pc, 1_dim);
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

    auto ta = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    auto tb = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<1, M / P>);
    auto pa = ntt::make_tensor<ntt::vector<float, P, P>>(
        ntt::fixed_shape_v<M / P, N / P>);
    auto pc = ntt::make_tensor<ntt::vector<float, P, P>>(
        ntt::fixed_shape_v<1, M / P, N / P>);
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
    std::iota(tb.elements().begin(), tb.elements().end(), 0);
    NttTest::init_tensor(pa, -10.f, 10.f);
    ntt::pack(ta, pa, ntt::fixed_shape_v<0, 1>);

    // warm up
    for (size_t i = 0; i < warmup_size; i++)
        ntt::gather(pa, tb, pc, 0_dim);

    // run
    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_size; i++) {
        ntt::gather(pa, tb, pc, 0_dim);
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
    auto ta = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    auto tb = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<1, N / P>);
    auto pa = ntt::make_tensor<ntt::vector<float, P, P>>(
        ntt::fixed_shape_v<M / P, N / P>);
    auto pc = ntt::make_tensor<ntt::vector<float, P, P>>(
        ntt::fixed_shape_v<M / P, 1, N / P>);
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
    std::iota(tb.elements().begin(), tb.elements().end(), 0);
    NttTest::init_tensor(pa, -10.f, 10.f);
    ntt::pack(ta, pa, ntt::fixed_shape_v<0, 1>);

    // warm up
    for (size_t i = 0; i < warmup_size; i++)
        ntt::gather(pa, tb, pc, 1_dim);

    // run
    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_size; i++) {
        ntt::gather(pa, tb, pc, 1_dim);
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