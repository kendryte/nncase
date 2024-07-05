#include "ntt_test.h"
#include <iomanip>
#include <nncase/ntt/ntt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <type_traits>

using namespace nncase;

int main() {

    constexpr size_t case_num = 100;
    // no pack
    {
        constexpr size_t M = 256;
        constexpr size_t K = 256;
        constexpr size_t N = 256;
        ntt::tensor<float, ntt::fixed_shape<M, K>> ta;
        ntt::tensor<float, ntt::fixed_shape<K, N>> tb;
        ntt::tensor<float, ntt::fixed_shape<M, N>> tc;
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        std::iota(tb.elements().begin(), tb.elements().end(), 0.f);

        ntt::matmul<false>(ta, tb, tc);
        auto t1 = NttTest::get_cpu_cycle();
        for (size_t i = 0; i < case_num; i++) {
            ntt::matmul<false>(ta, tb, tc);
        }
        auto t2 = NttTest::get_cpu_cycle();

        assert(std::abs(tc(M - 1, N - 1) - 551162314880.00) / 551162314880.00 <
               1e-6f);
        std::string module = "benchmark_ntt_matmul";
        std::string mode = "no_pack";
        std::cout << module << "_" << mode << " took " << std::setprecision(0)
                  << std::fixed << static_cast<float>(t2 - t1) / case_num
                  << " cycles" << std::endl;
    }

    // pack K
    {
        constexpr size_t M = 256;
        constexpr size_t K = 256;
        constexpr size_t N = 256;
        constexpr size_t P = 8;
        ntt::tensor<float, ntt::fixed_shape<M, K>> ta;
        ntt::tensor<float, ntt::fixed_shape<K, N>> tb;
        ntt::tensor<float, ntt::fixed_shape<M, N>> tc;
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        std::iota(tb.elements().begin(), tb.elements().end(), 0.f);
        alignas(32)
            ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M, K / P>>
                pa;
        alignas(32)
            ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<K / P, N>>
                pb;
        ntt::pack<1>(ta, pa);
        ntt::pack<0>(tb, pb);

        ntt::matmul<false>(pa, pb, tc, ntt::fixed_shape<1>{},
                           ntt::fixed_shape<0>{}, ntt::fixed_shape<0>{},
                           ntt::fixed_shape<0>{});
        auto t1 = NttTest::get_cpu_cycle();
        for (size_t i = 0; i < case_num; i++) {
            ntt::matmul<false>(pa, pb, tc, ntt::fixed_shape<1>{},
                               ntt::fixed_shape<0>{}, ntt::fixed_shape<0>{},
                               ntt::fixed_shape<0>{});
        }
        auto t2 = NttTest::get_cpu_cycle();

        assert(std::abs(tc(M - 1, N - 1) - 551162314880.f) < 1e-6f);
        std::string module = "benchmark_ntt_matmul";
        std::string mode = "pack_K";
        std::cout << module << "_" << mode << " took " << std::setprecision(0)
                  << std::fixed << static_cast<float>(t2 - t1) / case_num
                  << " cycles" << std::endl;
    }

    // pack M
    {
        constexpr size_t M = 256;
        constexpr size_t K = 256;
        constexpr size_t N = 256;
        constexpr size_t P = 8;
        ntt::tensor<float, ntt::fixed_shape<M, K>> ta;
        ntt::tensor<float, ntt::fixed_shape<K, N>> tb;
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        std::iota(tb.elements().begin(), tb.elements().end(), 0.f);
        alignas(32)
            ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M / P, K>>
                pa;
        alignas(32)
            ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M / P, N>>
                pc;
        ntt::pack<0>(ta, pa);
        ntt::matmul<false>(pa, tb, pc, ntt::fixed_shape<0>{},
                           ntt::fixed_shape<0>{}, ntt::fixed_shape<>{},
                           ntt::fixed_shape<0>{});
        auto t1 = NttTest::get_cpu_cycle();
        for (size_t i = 0; i < case_num; i++) {
            ntt::matmul<false>(pa, tb, pc, ntt::fixed_shape<0>{},
                               ntt::fixed_shape<0>{}, ntt::fixed_shape<>{},
                               ntt::fixed_shape<0>{});
        }
        auto t2 = NttTest::get_cpu_cycle();

        assert(std::abs(pc(31, 255)(7) - 551162314880.f) < 1e-6f);
        std::string module = "benchmark_ntt_matmul";
        std::string mode = "pack_M";
        std::cout << module << "_" << mode << " took " << std::setprecision(0)
                  << std::fixed << static_cast<float>(t2 - t1) / case_num
                  << " cycles" << std::endl;
    }

    // pack N
    {
        constexpr size_t M = 256;
        constexpr size_t K = 256;
        constexpr size_t N = 256;
        constexpr size_t P = 8;
        ntt::tensor<float, ntt::fixed_shape<M, K>> ta;
        ntt::tensor<float, ntt::fixed_shape<K, N>> tb;
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        std::iota(tb.elements().begin(), tb.elements().end(), 0.f);
        alignas(32)
            ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<K, N / P>>
                pb;
        alignas(32)
            ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M, N / P>>
                pc;
        ntt::pack<1>(tb, pb);
        ntt::matmul<false>(ta, pb, pc, ntt::fixed_shape<>{},
                           ntt::fixed_shape<0>{}, ntt::fixed_shape<1>{},
                           ntt::fixed_shape<0>{});
        auto t1 = NttTest::get_cpu_cycle();
        for (size_t i = 0; i < case_num; i++) {
            ntt::matmul<false>(ta, pb, pc, ntt::fixed_shape<>{},
                               ntt::fixed_shape<0>{}, ntt::fixed_shape<1>{},
                               ntt::fixed_shape<0>{});
        }
        auto t2 = NttTest::get_cpu_cycle();

        assert(std::abs(pc(255, 31)(7) - 551162314880.f) < 1e-6f);
        std::string module = "benchmark_ntt_matmul";
        std::string mode = "pack_N";
        std::cout << module << "_" << mode << " took " << std::setprecision(0)
                  << std::fixed << static_cast<float>(t2 - t1) / case_num
                  << " cycles" << std::endl;
    }

    // pack M and N
    {
        constexpr size_t M = 256;
        constexpr size_t K = 256;
        constexpr size_t N = 256;
        constexpr size_t P = 8;
        ntt::tensor<float, ntt::fixed_shape<M, K>> ta;
        ntt::tensor<float, ntt::fixed_shape<K, N>> tb;
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        std::iota(tb.elements().begin(), tb.elements().end(), 0.f);
        alignas(32)
            ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M / P, K>>
                pa;
        alignas(32)
            ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<K, N / P>>
                pb;
        alignas(32) ntt::tensor<ntt::vector<float, P, P>,
                                ntt::fixed_shape<M / P, N / P>>
            pc;
        ntt::pack<0>(ta, pa);
        ntt::pack<1>(tb, pb);
        ntt::matmul<false>(pa, pb, pc, ntt::fixed_shape<0>{},
                           ntt::fixed_shape<0>{}, ntt::fixed_shape<1>{},
                           ntt::fixed_shape<0>{});
        auto t1 = NttTest::get_cpu_cycle();
        for (size_t i = 0; i < case_num; i++) {
            ntt::matmul<false>(pa, pb, pc, ntt::fixed_shape<0>{},
                               ntt::fixed_shape<0>{}, ntt::fixed_shape<1>{},
                               ntt::fixed_shape<0>{});
        }
        auto t2 = NttTest::get_cpu_cycle();

        assert(std::abs(pc(31, 31)(7, 7) - 551162314880.f) < 1e-6f);
        std::string module = "benchmark_ntt_matmul";
        std::string mode = "pack_M_N";
        std::cout << module << "_" << mode << " took " << std::setprecision(0)
                  << std::fixed << static_cast<float>(t2 - t1) / case_num
                  << " cycles" << std::endl;
    }

    // pack M and K
    {
        constexpr size_t M = 256;
        constexpr size_t K = 256;
        constexpr size_t N = 256;
        constexpr size_t P = 8;
        ntt::tensor<float, ntt::fixed_shape<M, K>> ta;
        ntt::tensor<float, ntt::fixed_shape<K, N>> tb;
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        std::iota(tb.elements().begin(), tb.elements().end(), 0.f);
        alignas(32) ntt::tensor<ntt::vector<float, P, P>,
                                ntt::fixed_shape<M / P, K / P>>
            pa;
        alignas(32)
            ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<K / P, N>>
                pb;
        alignas(32)
            ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M / P, N>>
                pc;
        ntt::pack<0, 1>(ta, pa);
        ntt::pack<0>(tb, pb);
        ntt::matmul<false>(pa, pb, pc, ntt::fixed_shape<0, 1>{},
                           ntt::fixed_shape<0>{}, ntt::fixed_shape<0>{},
                           ntt::fixed_shape<0>{});
        auto t1 = NttTest::get_cpu_cycle();
        for (size_t i = 0; i < case_num; i++) {
            ntt::matmul<false>(pa, pb, pc, ntt::fixed_shape<0, 1>{},
                               ntt::fixed_shape<0>{}, ntt::fixed_shape<0>{},
                               ntt::fixed_shape<0>{});
        }
        auto t2 = NttTest::get_cpu_cycle();

        assert(std::abs(pc(31, 255)(7) - 551162314880.f) < 1e-6f);
        std::string module = "benchmark_ntt_matmul";
        std::string mode = "pack_M_K";
        std::cout << module << "_" << mode << " took " << std::setprecision(0)
                  << std::fixed << static_cast<float>(t2 - t1) / case_num
                  << " cycles" << std::endl;
    }

    // pack K and N
    {
        constexpr size_t M = 256;
        constexpr size_t K = 256;
        constexpr size_t N = 256;
        constexpr size_t P = 8;
        ntt::tensor<float, ntt::fixed_shape<M, K>> ta;
        ntt::tensor<float, ntt::fixed_shape<K, N>> tb;
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        std::iota(tb.elements().begin(), tb.elements().end(), 0.f);
        alignas(32)
            ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M, K / P>>
                pa;
        alignas(32) ntt::tensor<ntt::vector<float, P, P>,
                                ntt::fixed_shape<K / P, N / P>>
            pb;
        alignas(32)
            ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M, N / P>>
                pc;
        ntt::pack<1>(ta, pa);
        ntt::pack<0, 1>(tb, pb);
        ntt::matmul<false>(pa, pb, pc, ntt::fixed_shape<1>{},
                           ntt::fixed_shape<0>{}, ntt::fixed_shape<0, 1>{},
                           ntt::fixed_shape<0>{});

        auto t1 = NttTest::get_cpu_cycle();
        for (size_t i = 0; i < case_num; i++) {
            ntt::matmul<false>(pa, pb, pc, ntt::fixed_shape<1>{},
                               ntt::fixed_shape<0>{}, ntt::fixed_shape<0, 1>{},
                               ntt::fixed_shape<0>{});
        }
        auto t2 = NttTest::get_cpu_cycle();

        assert(std::abs(pc(255, 31)(7) - 551162314880.f) < 1e-6f);
        std::string module = "benchmark_ntt_matmul";
        std::string mode = "pack_K_N";
        std::cout << module << "_" << mode << " took " << std::setprecision(0)
                  << std::fixed << static_cast<float>(t2 - t1) / case_num
                  << " cycles" << std::endl;
    }

    // pack M, K and N
    {
        constexpr size_t M = 256;
        constexpr size_t K = 256;
        constexpr size_t N = 256;
        constexpr size_t P = 8;
        ntt::tensor<float, ntt::fixed_shape<M, K>> ta;
        ntt::tensor<float, ntt::fixed_shape<K, N>> tb;
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        std::iota(tb.elements().begin(), tb.elements().end(), 0.f);
        alignas(32) ntt::tensor<ntt::vector<float, P, P>,
                                ntt::fixed_shape<M / P, K / P>>
            pa;
        alignas(32) ntt::tensor<ntt::vector<float, P, P>,
                                ntt::fixed_shape<K / P, N / P>>
            pb;
        alignas(32) ntt::tensor<ntt::vector<float, P, P>,
                                ntt::fixed_shape<K / P, N / P>>
            pc;
        ntt::pack<0, 1>(ta, pa);
        ntt::pack<0, 1>(tb, pb);
        ntt::matmul<false>(pa, pb, pc, ntt::fixed_shape<0, 1>{},
                           ntt::fixed_shape<0>{}, ntt::fixed_shape<0, 1>{},
                           ntt::fixed_shape<0>{});

        auto t1 = NttTest::get_cpu_cycle();
        for (size_t i = 0; i < case_num; i++) {
            ntt::matmul<false>(pa, pb, pc, ntt::fixed_shape<0, 1>{},
                               ntt::fixed_shape<0>{}, ntt::fixed_shape<0, 1>{},
                               ntt::fixed_shape<0>{});
        }
        auto t2 = NttTest::get_cpu_cycle();

        assert(std::abs(pc(31, 31)(7, 7) - 551162314880.f) < 1e-6f);
        std::string module = "benchmark_ntt_matmul";
        std::string mode = "pack_M_K_N";
        std::cout << module << "_" << mode << " took " << std::setprecision(0)
                  << std::fixed << static_cast<float>(t2 - t1) / case_num
                  << " cycles" << std::endl;
    }

    return 0;
}