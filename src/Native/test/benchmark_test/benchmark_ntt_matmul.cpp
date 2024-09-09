#include "ntt_test.h"
#include <iomanip>
#include <nncase/ntt/ntt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <type_traits>

using namespace nncase;

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
    ntt::tensor<float, ntt::fixed_shape<M, K>> ta;
    ntt::tensor<float, ntt::fixed_shape<K, N>> tb;
    ntt::tensor<float, ntt::fixed_shape<M, N>> tc;
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
    std::iota(tb.elements().begin(), tb.elements().end(), 0.f);

    for (size_t i = 0; i < warmup_num; i++)
        ntt::matmul<false>(ta, tb, tc);

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::matmul<false>(ta, tb, tc);
    }
    auto t2 = NttTest::get_cpu_cycle();
    asm volatile("" ::"g"(tc));

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
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
    ntt::tensor<float, ntt::fixed_shape<M, K>> ta;
    ntt::tensor<float, ntt::fixed_shape<K, N>> tb;
    ntt::tensor<float, ntt::fixed_shape<M, N>> tc;
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
    std::iota(tb.elements().begin(), tb.elements().end(), 0.f);
    alignas(32) ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M, K / P>>
        pa;
    alignas(32) ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<K / P, N>>
        pb;
    ntt::pack<1>(ta, pa);
    ntt::pack<0>(tb, pb);

    for (size_t i = 0; i < warmup_num; i++)
        ntt::matmul<false>(pa, pb, tc, ntt::fixed_shape<1>{},
                           ntt::fixed_shape<0>{}, ntt::fixed_shape<0>{},
                           ntt::fixed_shape<0>{});

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::matmul<false>(pa, pb, tc, ntt::fixed_shape<1>{},
                           ntt::fixed_shape<0>{}, ntt::fixed_shape<0>{},
                           ntt::fixed_shape<0>{});
    }
    auto t2 = NttTest::get_cpu_cycle();
    asm volatile("" ::"g"(tc));

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
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
    ntt::tensor<float, ntt::fixed_shape<M, K>> ta;
    ntt::tensor<float, ntt::fixed_shape<K, N>> tb;
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
    std::iota(tb.elements().begin(), tb.elements().end(), 0.f);
    alignas(32) ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M / P, K>>
        pa;
    alignas(32) ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M / P, N>>
        pc;
    ntt::pack<0>(ta, pa);

    for (size_t i = 0; i < warmup_num; i++)
        ntt::matmul<false>(pa, tb, pc, ntt::fixed_shape<0>{},
                           ntt::fixed_shape<0>{}, ntt::fixed_shape<>{},
                           ntt::fixed_shape<0>{});

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::matmul<false>(pa, tb, pc, ntt::fixed_shape<0>{},
                           ntt::fixed_shape<0>{}, ntt::fixed_shape<>{},
                           ntt::fixed_shape<0>{});
    }
    auto t2 = NttTest::get_cpu_cycle();
    asm volatile("" ::"g"(pc));

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
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
    ntt::tensor<float, ntt::fixed_shape<M, K>> ta;
    ntt::tensor<float, ntt::fixed_shape<K, N>> tb;
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
    std::iota(tb.elements().begin(), tb.elements().end(), 0.f);
    alignas(32) ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<K, N / P>>
        pb;
    alignas(32) ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M, N / P>>
        pc;
    ntt::pack<1>(tb, pb);

    for (size_t i = 0; i < warmup_num; i++)
        ntt::matmul<false>(ta, pb, pc, ntt::fixed_shape<>{},
                           ntt::fixed_shape<0>{}, ntt::fixed_shape<1>{},
                           ntt::fixed_shape<0>{});

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::matmul<false>(ta, pb, pc, ntt::fixed_shape<>{},
                           ntt::fixed_shape<0>{}, ntt::fixed_shape<1>{},
                           ntt::fixed_shape<0>{});
    }
    auto t2 = NttTest::get_cpu_cycle();
    asm volatile("" ::"g"(pc));

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
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
    ntt::tensor<float, ntt::fixed_shape<M, K>> ta;
    ntt::tensor<float, ntt::fixed_shape<K, N>> tb;
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
    std::iota(tb.elements().begin(), tb.elements().end(), 0.f);
    alignas(32) ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M / P, K>>
        pa;
    alignas(32) ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<K, N / P>>
        pb;
    alignas(32)
        ntt::tensor<ntt::vector<float, P, P>, ntt::fixed_shape<M / P, N / P>>
            pc;
    ntt::pack<0>(ta, pa);
    ntt::pack<1>(tb, pb);

    for (size_t i = 0; i < warmup_num; i++)
        ntt::matmul<false>(pa, pb, pc, ntt::fixed_shape<0>{},
                           ntt::fixed_shape<0>{}, ntt::fixed_shape<1>{},
                           ntt::fixed_shape<0>{});

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::matmul<false>(pa, pb, pc, ntt::fixed_shape<0>{},
                           ntt::fixed_shape<0>{}, ntt::fixed_shape<1>{},
                           ntt::fixed_shape<0>{});
    }
    auto t2 = NttTest::get_cpu_cycle();
    asm volatile("" ::"g"(pc));

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
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
    ntt::tensor<float, ntt::fixed_shape<M, K>> ta;
    ntt::tensor<float, ntt::fixed_shape<K, N>> tb;
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
    std::iota(tb.elements().begin(), tb.elements().end(), 0.f);
    alignas(32)
        ntt::tensor<ntt::vector<float, P, P>, ntt::fixed_shape<M / P, K / P>>
            pa;
    alignas(32) ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<K / P, N>>
        pb;
    alignas(32) ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M / P, N>>
        pc;
    ntt::pack<0, 1>(ta, pa);
    ntt::pack<0>(tb, pb);

    for (size_t i = 0; i < warmup_num; i++)
        ntt::matmul<false>(pa, pb, pc, ntt::fixed_shape<0, 1>{},
                           ntt::fixed_shape<0>{}, ntt::fixed_shape<0>{},
                           ntt::fixed_shape<0>{});

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::matmul<false>(pa, pb, pc, ntt::fixed_shape<0, 1>{},
                           ntt::fixed_shape<0>{}, ntt::fixed_shape<0>{},
                           ntt::fixed_shape<0>{});
    }
    auto t2 = NttTest::get_cpu_cycle();
    asm volatile("" ::"g"(pc));

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
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
    ntt::tensor<float, ntt::fixed_shape<M, K>> ta;
    ntt::tensor<float, ntt::fixed_shape<K, N>> tb;
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
    std::iota(tb.elements().begin(), tb.elements().end(), 0.f);
    alignas(32) ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M, K / P>>
        pa;
    alignas(32)
        ntt::tensor<ntt::vector<float, P, P>, ntt::fixed_shape<K / P, N / P>>
            pb;
    alignas(32) ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M, N / P>>
        pc;
    ntt::pack<1>(ta, pa);
    ntt::pack<0, 1>(tb, pb);

    for (size_t i = 0; i < warmup_num; i++)
        ntt::matmul<false>(pa, pb, pc, ntt::fixed_shape<1>{},
                           ntt::fixed_shape<0>{}, ntt::fixed_shape<0, 1>{},
                           ntt::fixed_shape<0>{});

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::matmul<false>(pa, pb, pc, ntt::fixed_shape<1>{},
                           ntt::fixed_shape<0>{}, ntt::fixed_shape<0, 1>{},
                           ntt::fixed_shape<0>{});
    }
    auto t2 = NttTest::get_cpu_cycle();
    asm volatile("" ::"g"(pc));

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
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
    ntt::tensor<float, ntt::fixed_shape<M, K>> ta;
    ntt::tensor<float, ntt::fixed_shape<K, N>> tb;
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
    std::iota(tb.elements().begin(), tb.elements().end(), 0.f);
    alignas(32)
        ntt::tensor<ntt::vector<float, P, P>, ntt::fixed_shape<M / P, K / P>>
            pa;
    alignas(32)
        ntt::tensor<ntt::vector<float, P, P>, ntt::fixed_shape<K / P, N / P>>
            pb;
    alignas(32)
        ntt::tensor<ntt::vector<float, P, P>, ntt::fixed_shape<K / P, N / P>>
            pc;
    ntt::pack<0, 1>(ta, pa);
    ntt::pack<0, 1>(tb, pb);

    for (size_t i = 0; i < warmup_num; i++)
        ntt::matmul<false>(pa, pb, pc, ntt::fixed_shape<0, 1>{},
                           ntt::fixed_shape<0>{}, ntt::fixed_shape<0, 1>{},
                           ntt::fixed_shape<0>{});

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::matmul<false>(pa, pb, pc, ntt::fixed_shape<0, 1>{},
                           ntt::fixed_shape<0>{}, ntt::fixed_shape<0, 1>{},
                           ntt::fixed_shape<0>{});
    }
    auto t2 = NttTest::get_cpu_cycle();
    asm volatile("" ::"g"(pc));

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
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