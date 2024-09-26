#include "../../include/nncase/ntt/ukernels.h"
#include "ntt_test.h"
#include <iomanip>
#include <iostream>
#include <nncase/ntt/ntt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <type_traits>

using namespace nncase;

template <size_t M, size_t K, size_t N> void benchmark_ntt_matmul_pack_NONE() {
    constexpr size_t warmup_num = 10;
    constexpr size_t run_num = 3000;

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

    auto ops = M * N * (2 * K - 1);
    auto freq = 3.2 * 1e9;
    std::cout << (__FUNCTION__ + std::strlen("benchmark_ntt_matmul_pack_"))
              << std::setprecision(0) << std::fixed << ", M:" << M
              << ", K:" << K << ", N:" << N
              << ", Cycles:" << static_cast<float>(t2 - t1) / run_num
              << ", GFLOPS:"
              << ops / (static_cast<float>(t2 - t1) / run_num / freq) * 1e-9
              << std::endl;
}

// pack K
template <size_t M, size_t K, size_t N> void benchmark_ntt_matmul_pack_K() {
    constexpr size_t warmup_num = 10;
    constexpr size_t run_num = 3000;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

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

    auto ops = M * N * (2 * K - 1);
    auto freq = 3.2 * 1e9;
    std::cout << (__FUNCTION__ + std::strlen("benchmark_ntt_matmul_pack_"))
              << std::setprecision(0) << std::fixed << ", M:" << M
              << ", K:" << K / P << ", N:" << N
              << ", Cycles:" << static_cast<float>(t2 - t1) / run_num
              << ", GFLOPS:"
              << ops / (static_cast<float>(t2 - t1) / run_num / freq) * 1e-9
              << std::endl;
}

// pack M
template <size_t M, size_t K, size_t N> void benchmark_ntt_matmul_pack_M() {
    constexpr size_t warmup_num = 10;
    constexpr size_t run_num = 3000;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

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

    auto ops = M * N * (2 * K - 1);
    auto freq = 3.2 * 1e9;
    std::cout << (__FUNCTION__ + std::strlen("benchmark_ntt_matmul_pack_"))
              << std::setprecision(0) << std::fixed << ", M:" << M / P
              << ", K:" << K << ", N:" << N
              << ", Cycles:" << static_cast<float>(t2 - t1) / run_num
              << ", GFLOPS:"
              << ops / (static_cast<float>(t2 - t1) / run_num / freq) * 1e-9
              << std::endl;
}

// pack N
template <size_t M, size_t K, size_t N> void benchmark_ntt_matmul_pack_N() {
    constexpr size_t warmup_num = 10;
    constexpr size_t run_num = 3000;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

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

    auto ops = M * N * (2 * K - 1);
    auto freq = 3.2 * 1e9;
    std::cout << (__FUNCTION__ + std::strlen("benchmark_ntt_matmul_pack_"))
              << std::setprecision(0) << std::fixed << ", M:" << M
              << ", K:" << K << ", N:" << N / P
              << ", Cycles:" << static_cast<float>(t2 - t1) / run_num
              << ", GFLOPS:"
              << ops / (static_cast<float>(t2 - t1) / run_num / freq) * 1e-9
              << std::endl;
}

// pack M and N
template <size_t M, size_t K, size_t N> void benchmark_ntt_matmul_pack_M_N() {
    constexpr size_t warmup_num = 10;
    constexpr size_t run_num = 3000;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

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

    auto ops = M * N * (2 * K - 1);
    auto freq = 3.2 * 1e9;
    std::cout << (__FUNCTION__ + std::strlen("benchmark_ntt_matmul_pack_"))
              << std::setprecision(0) << std::fixed << ", M:" << M / P
              << ", K:" << K << ", N:" << N / P
              << ", Cycles:" << static_cast<float>(t2 - t1) / run_num
              << ", GFLOPS:"
              << ops / (static_cast<float>(t2 - t1) / run_num / freq) * 1e-9
              << std::endl;
}

// pack M and K
template <size_t M, size_t K, size_t N> void benchmark_ntt_matmul_pack_M_K() {
    constexpr size_t warmup_num = 10;
    constexpr size_t run_num = 3000;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

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

    auto ops = M * N * (2 * K - 1);
    auto freq = 3.2 * 1e9;
    std::cout << (__FUNCTION__ + std::strlen("benchmark_ntt_matmul_pack_"))
              << std::setprecision(0) << std::fixed << ", M:" << M / P
              << ", K:" << K / P << ", N:" << N
              << ", Cycles:" << static_cast<float>(t2 - t1) / run_num
              << ", GFLOPS:"
              << ops / (static_cast<float>(t2 - t1) / run_num / freq) * 1e-9
              << std::endl;
}

// pack K and N
template <size_t M, size_t K, size_t N> void benchmark_ntt_matmul_pack_K_N() {
    constexpr size_t warmup_num = 10;
    constexpr size_t run_num = 3000;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

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

    auto ops = M * N * (2 * K - 1);
    auto freq = 3.2 * 1e9;
    std::cout << (__FUNCTION__ + std::strlen("benchmark_ntt_matmul_pack_"))
              << std::setprecision(0) << std::fixed << ", M:" << M
              << ", K:" << K / P << ", N:" << N / P
              << ", Cycles:" << static_cast<float>(t2 - t1) / run_num
              << ", GFLOPS:"
              << ops / (static_cast<float>(t2 - t1) / run_num / freq) * 1e-9
              << std::endl;
}

// pack M, K and N
template <size_t M, size_t K, size_t N> void benchmark_ntt_matmul_pack_M_K_N() {
    constexpr size_t warmup_num = 10;
    constexpr size_t run_num = 3000;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

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

    auto ops = M * N * (2 * K - 1);
    auto freq = 3.2 * 1e9;
    std::cout << (__FUNCTION__ + std::strlen("benchmark_ntt_matmul_pack_"))
              << std::setprecision(0) << std::fixed << ", M:" << M / P
              << ", K:" << K / P << ", N:" << N / P
              << ", Cycles:" << static_cast<float>(t2 - t1) / run_num
              << ", GFLOPS:"
              << ops / (static_cast<float>(t2 - t1) / run_num / freq) * 1e-9
              << std::endl;
}

#define BENCHMARK_NTT_MATMUL(MODE, M_BASE, K_BASE, N_BASE, M_TILE, N_TILE)     \
    benchmark_ntt_matmul_pack_##MODE<M_BASE * 1 * M_TILE, K_BASE * 1,          \
                                     N_BASE * 1 * N_TILE>();                   \
    benchmark_ntt_matmul_pack_##MODE<M_BASE * 2 * M_TILE, K_BASE * 2,          \
                                     N_BASE * 2 * N_TILE>();                   \
    benchmark_ntt_matmul_pack_##MODE<M_BASE * 3 * M_TILE, K_BASE * 3,          \
                                     N_BASE * 3 * N_TILE>();                   \
    benchmark_ntt_matmul_pack_##MODE<M_BASE * 4 * M_TILE, K_BASE * 4,          \
                                     N_BASE * 4 * N_TILE>();                   \
    benchmark_ntt_matmul_pack_##MODE<M_BASE * 5 * M_TILE, K_BASE * 5,          \
                                     N_BASE * 5 * N_TILE>();                   \
    benchmark_ntt_matmul_pack_##MODE<M_BASE * 6 * M_TILE, K_BASE * 6,          \
                                     N_BASE * 6 * N_TILE>();                   \
    benchmark_ntt_matmul_pack_##MODE<M_BASE * 7 * M_TILE, K_BASE * 7,          \
                                     N_BASE * 7 * N_TILE>();                   \
    benchmark_ntt_matmul_pack_##MODE<M_BASE * 8 * M_TILE, K_BASE * 8,          \
                                     N_BASE * 8 * N_TILE>();                   \
    benchmark_ntt_matmul_pack_##MODE<M_BASE * 9 * M_TILE, K_BASE * 9,          \
                                     N_BASE * 9 * N_TILE>();                   \
    benchmark_ntt_matmul_pack_##MODE<M_BASE * 10 * M_TILE, K_BASE * 10,        \
                                     N_BASE * 10 * N_TILE>();                  \
    benchmark_ntt_matmul_pack_##MODE<M_BASE * 11 * M_TILE, K_BASE * 11,        \
                                     N_BASE * 11 * N_TILE>();                  \
    benchmark_ntt_matmul_pack_##MODE<M_BASE * 12 * M_TILE, K_BASE * 12,        \
                                     N_BASE * 12 * N_TILE>();                  \
    benchmark_ntt_matmul_pack_##MODE<M_BASE * 13 * M_TILE, K_BASE * 13,        \
                                     N_BASE * 13 * N_TILE>();                  \
    benchmark_ntt_matmul_pack_##MODE<M_BASE * 14 * M_TILE, K_BASE * 14,        \
                                     N_BASE * 14 * N_TILE>();                  \
    benchmark_ntt_matmul_pack_##MODE<M_BASE * 15 * M_TILE, K_BASE * 15,        \
                                     N_BASE * 15 * N_TILE>();                  \
    benchmark_ntt_matmul_pack_##MODE<M_BASE * 16 * M_TILE, K_BASE * 16,        \
                                     N_BASE * 16 * N_TILE>();

template <nncase::ntt::ukernels::mamtul_pack_kind PackKind>
void matmul_primitive_analysis() {

    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    switch (PackKind) {
    case ntt::ukernels::mamtul_pack_kind::no_pack:
        // std::cout << "No packing" << std::endl;
        break;
    case ntt::ukernels::mamtul_pack_kind::pack_m:

        // std::cout << "Packing M" << std::endl;
        {
            using TLhsElem = ntt::vector<float, P>;
            using TRhsElem = float;
            using TOutElem = ntt::vector<float, P>;
            using policy_t =
                ntt::ukernels::u_matmul_policy<PackKind, TLhsElem, TRhsElem,
                                               TOutElem, true>;

            BENCHMARK_NTT_MATMUL(M, 8, 1, 1, policy_t::m0_tile,
                                 policy_t::n0_tile);
        }

        break;
    case ntt::ukernels::mamtul_pack_kind::pack_k:
        // std::cout << "Packing K" << std::endl;

        {
            using TLhsElem = ntt::vector<float, P>;
            using TRhsElem = ntt::vector<float, P>;
            using TOutElem = float;
            using policy_t =
                ntt::ukernels::u_matmul_policy<PackKind, TLhsElem, TRhsElem,
                                               TOutElem, true>;

            BENCHMARK_NTT_MATMUL(K, 1, 8, 1, policy_t::m0_tile,
                                 policy_t::n0_tile);
        }
        break;
    case ntt::ukernels::mamtul_pack_kind::pack_n:
        // std::cout << "Packing N" << std::endl;

        {
            using TLhsElem = float;
            using TRhsElem = ntt::vector<float, P>;
            using TOutElem = ntt::vector<float, P>;
            using policy_t =
                ntt::ukernels::u_matmul_policy<PackKind, TLhsElem, TRhsElem,
                                               TOutElem, true>;

            BENCHMARK_NTT_MATMUL(N, 1, 1, 8, policy_t::m0_tile,
                                 policy_t::n0_tile);
        }
        break;
    case ntt::ukernels::mamtul_pack_kind::pack_mn:
        // std::cout << "Packing M and N" << std::endl;

        {
            using TLhsElem = ntt::vector<float, P>;
            using TRhsElem = ntt::vector<float, P>;
            using TOutElem = ntt::vector<float, P, P>;
            using policy_t =
                ntt::ukernels::u_matmul_policy<PackKind, TLhsElem, TRhsElem,
                                               TOutElem, true>;

            BENCHMARK_NTT_MATMUL(M_N, 8, 1, 8, policy_t::m0_tile,
                                 policy_t::n0_tile);
        }
        break;
    case ntt::ukernels::mamtul_pack_kind::pack_mk:
        // std::cout << "Packing M and K" << std::endl;

        {
            using TLhsElem = ntt::vector<float, P, P>;
            using TRhsElem = ntt::vector<float, P>;
            using TOutElem = ntt::vector<float, P>;
            using policy_t =
                ntt::ukernels::u_matmul_policy<PackKind, TLhsElem, TRhsElem,
                                               TOutElem, true>;

            BENCHMARK_NTT_MATMUL(M_K, 8, 8, 1, policy_t::m0_tile,
                                 policy_t::n0_tile);
        }
        break;
    case ntt::ukernels::mamtul_pack_kind::pack_kn:
        // std::cout << "Packing K and N" << std::endl;

        {
            using TLhsElem = ntt::vector<float, P>;
            using TRhsElem = ntt::vector<float, P, P>;
            using TOutElem = ntt::vector<float, P>;
            using policy_t =
                ntt::ukernels::u_matmul_policy<PackKind, TLhsElem, TRhsElem,
                                               TOutElem, true>;

            BENCHMARK_NTT_MATMUL(K_N, 1, 8, 8, policy_t::m0_tile,
                                 policy_t::n0_tile);
        }
        break;
    case ntt::ukernels::mamtul_pack_kind::pack_mkn:
        // std::cout << "Packing M, K, and N" << std::endl;

        {
            using TLhsElem = ntt::vector<float, P, P>;
            using TRhsElem = ntt::vector<float, P, P>;
            using TOutElem = ntt::vector<float, P, P>;
            using policy_t =
                ntt::ukernels::u_matmul_policy<PackKind, TLhsElem, TRhsElem,
                                               TOutElem, true>;

            BENCHMARK_NTT_MATMUL(M_K_N, 8, 8, 8, policy_t::m0_tile,
                                 policy_t::n0_tile);
        }
        break;
    default:
        std::cout << "Invalid packing kind" << std::endl;
        break;
    }
}

int main() {

#if __x86_64__

    {
        const auto PackMode = nncase::ntt::ukernels::mamtul_pack_kind::pack_m;
        matmul_primitive_analysis<PackMode>();
    }

    {
        const auto PackMode = nncase::ntt::ukernels::mamtul_pack_kind::pack_k;
        matmul_primitive_analysis<PackMode>();
    }

    {
        const auto PackMode = nncase::ntt::ukernels::mamtul_pack_kind::pack_n;
        matmul_primitive_analysis<PackMode>();
    }

    {
        const auto PackMode = nncase::ntt::ukernels::mamtul_pack_kind::pack_mn;
        matmul_primitive_analysis<PackMode>();
    }

    {
        const auto PackMode = nncase::ntt::ukernels::mamtul_pack_kind::pack_mk;
        matmul_primitive_analysis<PackMode>();
    }

    {
        const auto PackMode = nncase::ntt::ukernels::mamtul_pack_kind::pack_kn;
        matmul_primitive_analysis<PackMode>();
    }

    {
        const auto PackMode = nncase::ntt::ukernels::mamtul_pack_kind::pack_mkn;
        matmul_primitive_analysis<PackMode>();
    }

#endif

    return 0;
}