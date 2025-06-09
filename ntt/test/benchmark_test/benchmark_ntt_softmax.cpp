#include "ntt_test.h"
#include <iomanip>
#include <nncase/ntt/ntt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <type_traits>

using namespace nncase;

// no pack
void benchmark_ntt_softmax_reduceAxis1_noPack() {
    constexpr size_t warmup_num = 10;
    constexpr size_t run_num = 3000;
#if __riscv
    constexpr size_t D0 = 3;
    constexpr size_t D1 = 16;
    constexpr size_t D2 = 16;
#else
    constexpr size_t D0 = 3;
    constexpr size_t D1 = 16;
    constexpr size_t D2 = 16;
#endif
    ntt::tensor<float, ntt::fixed_shape<D0, D1, D2>> buffer_1;
    NttTest::init_tensor(buffer_1, -10.f, 10.f);

    ntt::tensor<float, ntt::fixed_shape<D0, D1, D2>> ntt_output;

    for (size_t i = 0; i < warmup_num; i++)
        packed_softmax<1>(buffer_1, ntt_output, ntt::fixed_shape<>{});

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        packed_softmax<1>(buffer_1, ntt_output, ntt::fixed_shape<>{});
        asm volatile("" ::"g"(ntt_output));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

void benchmark_ntt_softmax_reduceAxis2_noPack() {
    constexpr size_t warmup_num = 10;
    constexpr size_t run_num = 3000;
#if __riscv
    constexpr size_t D0 = 3;
    constexpr size_t D1 = 16;
    constexpr size_t D2 = 16;
#else
    constexpr size_t D0 = 3;
    constexpr size_t D1 = 16;
    constexpr size_t D2 = 16;
#endif
    ntt::tensor<float, ntt::fixed_shape<D0, D1, D2>> buffer_1;
    NttTest::init_tensor(buffer_1, -10.f, 10.f);

    ntt::tensor<float, ntt::fixed_shape<D0, D1, D2>> ntt_output;

    for (size_t i = 0; i < warmup_num; i++)
        packed_softmax<2>(buffer_1, ntt_output, ntt::fixed_shape<>{});

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        packed_softmax<2>(buffer_1, ntt_output, ntt::fixed_shape<>{});
        asm volatile("" ::"g"(ntt_output));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

void benchmark_ntt_softmax_reduceAxis1_packAxis1() {
    constexpr size_t warmup_num = 10;
    constexpr size_t run_num = 3000;
#if __riscv
    constexpr size_t D0 = 3;
    constexpr size_t D1 = 16;
    constexpr size_t D2 = 16;
#else
    constexpr size_t D0 = 3;
    constexpr size_t D1 = 16;
    constexpr size_t D2 = 16;
#endif
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    ntt::tensor<float, ntt::fixed_shape<D0, D1, D2>> buffer_1;
    NttTest::init_tensor(buffer_1, -10.f, 10.f);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<D0, D1 / P, D2>>
        buffer_2;

    pack<1>(buffer_1, buffer_2);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<D0, D1 / P, D2>>
        buffer_3;

    for (size_t i = 0; i < warmup_num; i++)
        packed_softmax<1>(buffer_2, buffer_3, ntt::fixed_shape<1>{});

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        packed_softmax<1>(buffer_2, buffer_3, ntt::fixed_shape<1>{});
        asm volatile("" ::"g"(buffer_3));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

void benchmark_ntt_softmax_reduceAxis2_packAxis2() {
    constexpr size_t warmup_num = 10;
    constexpr size_t run_num = 3000;
#if __riscv
    constexpr size_t D0 = 3;
    constexpr size_t D1 = 16;
    constexpr size_t D2 = 16;
#else
    constexpr size_t D0 = 3;
    constexpr size_t D1 = 16;
    constexpr size_t D2 = 16;
#endif
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    ntt::tensor<float, ntt::fixed_shape<D0, D1, D2>> buffer_1;
    NttTest::init_tensor(buffer_1, -10.f, 10.f);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<D0, D1, D2 / P>>
        buffer_2;

    pack<2>(buffer_1, buffer_2);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<D0, D1, D2 / P>>
        buffer_3;

    for (size_t i = 0; i < warmup_num; i++)
        packed_softmax<1>(buffer_2, buffer_3, ntt::fixed_shape<1>{});

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        packed_softmax<1>(buffer_2, buffer_3, ntt::fixed_shape<1>{});
        asm volatile("" ::"g"(buffer_3));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

void benchmark_ntt_softmax_reduceAxis2_packAxis1() {
    constexpr size_t warmup_num = 10;
    constexpr size_t run_num = 3000;
#if __riscv
    constexpr size_t D0 = 3;
    constexpr size_t D1 = 16;
    constexpr size_t D2 = 16;
#else
    constexpr size_t D0 = 3;
    constexpr size_t D1 = 16;
    constexpr size_t D2 = 16;
#endif
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    ntt::tensor<float, ntt::fixed_shape<D0, D1, D2>> buffer_1;
    NttTest::init_tensor(buffer_1, -10.f, 10.f);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<D0, D1 / P, D2>>
        buffer_2;

    pack<1>(buffer_1, buffer_2);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<D0, D1 / P, D2>>
        buffer_3;

    for (size_t i = 0; i < warmup_num; i++)
        packed_softmax<2>(buffer_2, buffer_3, ntt::fixed_shape<3>{});

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        packed_softmax<2>(buffer_2, buffer_3, ntt::fixed_shape<3>{});
        asm volatile("" ::"g"(buffer_3));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

void benchmark_ntt_softmax_reduceAxis1_packAxis2() {
    constexpr size_t warmup_num = 10;
    constexpr size_t run_num = 3000;
#if __riscv
    constexpr size_t D0 = 3;
    constexpr size_t D1 = 16;
    constexpr size_t D2 = 16;
#else
    constexpr size_t D0 = 3;
    constexpr size_t D1 = 16;
    constexpr size_t D2 = 16;
#endif
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    ntt::tensor<float, ntt::fixed_shape<D0, D1, D2>> buffer_1;
    NttTest::init_tensor(buffer_1, -10.f, 10.f);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<D0, D1, D2 / P>>
        buffer_2;

    pack<2>(buffer_1, buffer_2);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<D0, D1, D2 / P>>
        buffer_3;

    for (size_t i = 0; i < warmup_num; i++)
        packed_softmax<1>(buffer_2, buffer_3, ntt::fixed_shape<2>{});

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        packed_softmax<1>(buffer_2, buffer_3, ntt::fixed_shape<2>{});
        asm volatile("" ::"g"(buffer_3));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

int main() {

    benchmark_ntt_softmax_reduceAxis1_noPack();
    benchmark_ntt_softmax_reduceAxis2_noPack();
    benchmark_ntt_softmax_reduceAxis1_packAxis1();
    benchmark_ntt_softmax_reduceAxis2_packAxis2();
    benchmark_ntt_softmax_reduceAxis2_packAxis1();
    benchmark_ntt_softmax_reduceAxis1_packAxis2();

    return 0;
}