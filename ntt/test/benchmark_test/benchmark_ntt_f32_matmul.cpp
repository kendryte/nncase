#include "ntt_test.h"
#include <iomanip>
#include <nncase/ntt/ntt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <type_traits>

using namespace nncase;

#define MATMUL_INPUTS_INIT                                                     \
    [[maybe_unused]] auto ta =                                                 \
        ntt::make_tensor<float>(ntt::fixed_shape_v<M, K>);                     \
    [[maybe_unused]] auto tb =                                                 \
        ntt::make_tensor<float>(ntt::fixed_shape_v<K, N>);                     \
    [[maybe_unused]] auto tc =                                                 \
        ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);                     \
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);                \
    std::iota(tb.elements().begin(), tb.elements().end(), 0.f);

#define PRINT_FUNC_TICK                                                        \
    std::cout << __FUNCTION__ << " took " << std::setprecision(0)              \
              << std::fixed << static_cast<float>(t2 - t1) / run_num           \
              << " cycles" << std::endl;

// no pack
void benchmark_ntt_matmul_no_pack() {
    constexpr size_t warmup_num = 10;
    constexpr size_t run_num = 3000;
#if __riscv
    constexpr size_t M = 32;
    constexpr size_t K = 32;
    constexpr size_t N = 32;
#else
    constexpr size_t M = 32;
    constexpr size_t K = 32;
    constexpr size_t N = 32;
#endif
    MATMUL_INPUTS_INIT

    for (size_t i = 0; i < warmup_num; i++)
        ntt::matmul<false>(ta, tb, tc);

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::matmul<false>(ta, tb, tc);
        asm volatile("" ::"g"(tc));
    }
    auto t2 = NttTest::get_cpu_cycle();

    PRINT_FUNC_TICK
}

// pack K
void benchmark_ntt_matmul_pack_K() {
    constexpr size_t warmup_num = 10;
    constexpr size_t run_num = 3000;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
#if __riscv
    constexpr size_t M = 32;
    constexpr size_t K = 32;
    constexpr size_t N = 32;
#else
    constexpr size_t M = 32;
    constexpr size_t K = 32;
    constexpr size_t N = 32;
#endif
    MATMUL_INPUTS_INIT
    auto pa =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<M, K / P>);
    auto pb =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<K / P, N>);
    ntt::pack(ta, pa, ntt::fixed_shape_v<1>);
    ntt::pack(tb, pb, ntt::fixed_shape_v<0>);

    for (size_t i = 0; i < warmup_num; i++)
        ntt::matmul<false>(pa, pb, tc, ntt::fixed_shape_v<1>,
                           ntt::fixed_shape_v<>, ntt::fixed_shape_v<0>,
                           ntt::fixed_shape_v<>);

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::matmul<false>(pa, pb, tc, ntt::fixed_shape_v<1>,
                           ntt::fixed_shape_v<>, ntt::fixed_shape_v<0>,
                           ntt::fixed_shape_v<>);
        asm volatile("" ::"g"(tc));
    }
    auto t2 = NttTest::get_cpu_cycle();

    PRINT_FUNC_TICK
}

// pack M
void benchmark_ntt_matmul_pack_M() {
    constexpr size_t warmup_num = 10;
    constexpr size_t run_num = 3000;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
#if __riscv
    constexpr size_t M = 32;
    constexpr size_t K = 32;
    constexpr size_t N = 32;
#else
    constexpr size_t M = 32;
    constexpr size_t K = 32;
    constexpr size_t N = 32;
#endif
    MATMUL_INPUTS_INIT
    auto pa =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<M / P, K>);
    auto pc =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<M / P, N>);
    ntt::pack(ta, pa, ntt::fixed_shape_v<0>);

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::matmul<false>(pa, tb, pc, ntt::fixed_shape_v<0>,
                           ntt::fixed_shape_v<>, ntt::fixed_shape_v<>,
                           ntt::fixed_shape_v<>);
        asm volatile("" ::"g"(pc));
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::matmul<false>(pa, tb, pc, ntt::fixed_shape_v<0>,
                           ntt::fixed_shape_v<>, ntt::fixed_shape_v<>,
                           ntt::fixed_shape_v<>);
        asm volatile("" ::"g"(pc));
    }
    auto t2 = NttTest::get_cpu_cycle();

    PRINT_FUNC_TICK
}

// pack N
void benchmark_ntt_matmul_pack_N() {
    constexpr size_t warmup_num = 10;
    constexpr size_t run_num = 3000;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
#if __riscv
    constexpr size_t M = 32;
    constexpr size_t K = 32;
    constexpr size_t N = 32;
#else
    constexpr size_t M = 32;
    constexpr size_t K = 32;
    constexpr size_t N = 32;
#endif
    MATMUL_INPUTS_INIT
    auto pb =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<K, N / P>);
    auto pc =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<M, N / P>);
    ntt::pack(tb, pb, ntt::fixed_shape_v<1>);

    for (size_t i = 0; i < warmup_num; i++)
        ntt::matmul<false>(ta, pb, pc, ntt::fixed_shape_v<>,
                           ntt::fixed_shape_v<>, ntt::fixed_shape_v<1>,
                           ntt::fixed_shape_v<>);

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::matmul<false>(ta, pb, pc, ntt::fixed_shape_v<>,
                           ntt::fixed_shape_v<>, ntt::fixed_shape_v<1>,
                           ntt::fixed_shape_v<>);
        asm volatile("" ::"g"(pc));
    }
    auto t2 = NttTest::get_cpu_cycle();

    PRINT_FUNC_TICK
}

// pack M and N
void benchmark_ntt_matmul_pack_M_N() {
    constexpr size_t warmup_num = 10;
    constexpr size_t run_num = 3000;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
#if __riscv
    constexpr size_t M = 32;
    constexpr size_t K = 32;
    constexpr size_t N = 32;
#else
    constexpr size_t M = 32;
    constexpr size_t K = 32;
    constexpr size_t N = 32;
#endif
    MATMUL_INPUTS_INIT
    auto pa =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<M / P, K>);
    auto pb =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<K, N / P>);
    auto pc = ntt::make_tensor<ntt::vector<float, P, P>>(
        ntt::fixed_shape_v<M / P, N / P>);
    ntt::pack(ta, pa, ntt::fixed_shape_v<0>);
    ntt::pack(tb, pb, ntt::fixed_shape_v<1>);

    for (size_t i = 0; i < warmup_num; i++)
        ntt::matmul<false>(pa, pb, pc, ntt::fixed_shape_v<0>,
                           ntt::fixed_shape_v<>, ntt::fixed_shape_v<1>,
                           ntt::fixed_shape_v<>);

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::matmul<false>(pa, pb, pc, ntt::fixed_shape_v<0>,
                           ntt::fixed_shape_v<>, ntt::fixed_shape_v<1>,
                           ntt::fixed_shape_v<>);
        asm volatile("" ::"g"(pc));
    }
    auto t2 = NttTest::get_cpu_cycle();

    PRINT_FUNC_TICK
}

// pack M and K
void benchmark_ntt_matmul_pack_M_K() {
    constexpr size_t warmup_num = 10;
    constexpr size_t run_num = 3000;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
#if __riscv
    constexpr size_t M = 32;
    constexpr size_t K = 32;
    constexpr size_t N = 32;
#else
    constexpr size_t M = 32;
    constexpr size_t K = 32;
    constexpr size_t N = 32;
#endif
    MATMUL_INPUTS_INIT
    auto pa = ntt::make_tensor<ntt::vector<float, P, P>>(
        ntt::fixed_shape_v<M / P, K / P>);
    auto pb =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<K / P, N>);
    auto pc =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<M / P, N>);
    ntt::pack(ta, pa, ntt::fixed_shape_v<0, 1>);
    ntt::pack(tb, pb, ntt::fixed_shape_v<0>);

    for (size_t i = 0; i < warmup_num; i++)
        ntt::matmul<false>(pa, pb, pc, ntt::fixed_shape_v<0, 1>,
                           ntt::fixed_shape_v<>, ntt::fixed_shape_v<0>,
                           ntt::fixed_shape_v<>);

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::matmul<false>(pa, pb, pc, ntt::fixed_shape_v<0, 1>,
                           ntt::fixed_shape_v<>, ntt::fixed_shape_v<0>,
                           ntt::fixed_shape_v<>);
        asm volatile("" ::"g"(pc));
    }
    auto t2 = NttTest::get_cpu_cycle();

    PRINT_FUNC_TICK
}

// pack K and N
void benchmark_ntt_matmul_pack_K_N() {
    constexpr size_t warmup_num = 10;
    constexpr size_t run_num = 3000;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
#if __riscv
    constexpr size_t M = 32;
    constexpr size_t K = 32;
    constexpr size_t N = 32;
#else
    constexpr size_t M = 32;
    constexpr size_t K = 32;
    constexpr size_t N = 32;
#endif
    MATMUL_INPUTS_INIT
    auto pa =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<M, K / P>);
    auto pb = ntt::make_tensor<ntt::vector<float, P, P>>(
        ntt::fixed_shape_v<K / P, N / P>);
    auto pc =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<M, N / P>);
    ntt::pack(ta, pa, ntt::fixed_shape_v<1>);
    ntt::pack(tb, pb, ntt::fixed_shape_v<0, 1>);

    for (size_t i = 0; i < warmup_num; i++)
        ntt::matmul<false>(pa, pb, pc, ntt::fixed_shape_v<1>,
                           ntt::fixed_shape_v<>, ntt::fixed_shape_v<0, 1>,
                           ntt::fixed_shape_v<>);

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::matmul<false>(pa, pb, pc, ntt::fixed_shape_v<1>,
                           ntt::fixed_shape_v<>, ntt::fixed_shape_v<0, 1>,
                           ntt::fixed_shape_v<>);
        asm volatile("" ::"g"(pc));
    }
    auto t2 = NttTest::get_cpu_cycle();

    PRINT_FUNC_TICK
}

// pack M, K and N
void benchmark_ntt_matmul_pack_M_K_N() {
    constexpr size_t warmup_num = 10;
    constexpr size_t run_num = 3000;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
#if __riscv
    constexpr size_t M = 32;
    constexpr size_t K = 32;
    constexpr size_t N = 32;
#else
    constexpr size_t M = 32;
    constexpr size_t K = 32;
    constexpr size_t N = 32;
#endif
    MATMUL_INPUTS_INIT
    auto pa = ntt::make_tensor<ntt::vector<float, P, P>>(
        ntt::fixed_shape_v<M / P, K / P>);
    auto pb = ntt::make_tensor<ntt::vector<float, P, P>>(
        ntt::fixed_shape_v<K / P, N / P>);
    auto pc = ntt::make_tensor<ntt::vector<float, P, P>>(
        ntt::fixed_shape_v<M / P, N / P>);
    ntt::pack(ta, pa, ntt::fixed_shape_v<0, 1>);
    ntt::pack(tb, pb, ntt::fixed_shape_v<0, 1>);

    for (size_t i = 0; i < warmup_num; i++)
        ntt::matmul<false>(pa, pb, pc, ntt::fixed_shape_v<0, 1>,
                           ntt::fixed_shape_v<>, ntt::fixed_shape_v<0, 1>,
                           ntt::fixed_shape_v<>);

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::matmul<false>(pa, pb, pc, ntt::fixed_shape_v<0, 1>,
                           ntt::fixed_shape_v<>, ntt::fixed_shape_v<0, 1>,
                           ntt::fixed_shape_v<>);
        asm volatile("" ::"g"(pc));
    }
    auto t2 = NttTest::get_cpu_cycle();

    PRINT_FUNC_TICK
}

int main() {
    benchmark_ntt_matmul_no_pack();
    benchmark_ntt_matmul_pack_K();
    benchmark_ntt_matmul_pack_M();
    benchmark_ntt_matmul_pack_N();
    benchmark_ntt_matmul_pack_M_N();
    benchmark_ntt_matmul_pack_M_K();
    benchmark_ntt_matmul_pack_K_N();
    benchmark_ntt_matmul_pack_M_K_N();

    return 0;
}