#include "ntt_test.h"
#include <iomanip>
#include <nncase/ntt/ntt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <type_traits>

using namespace nncase;

// no vectorize
void benchmark_ntt_softmax_fixed_reduceAxis1_noVectorize() {
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
    auto buffer_1 = ntt::make_tensor<float>(ntt::fixed_shape_v<D0, D1, D2>);
    NttTest::init_tensor(buffer_1, -10.f, 10.f);

    auto ntt_output = ntt::make_tensor<float>(ntt::fixed_shape_v<D0, D1, D2>);

    for (size_t i = 0; i < warmup_num; i++)
        vectorized_softmax(buffer_1, ntt_output, 1_dim, ntt::fixed_shape_v<>);

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        vectorized_softmax(buffer_1, ntt_output, 1_dim, ntt::fixed_shape_v<>);
        asm volatile("" ::"g"(ntt_output));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

void benchmark_ntt_softmax_fixed_reduceAxis2_noVectorize() {
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
    auto buffer_1 = ntt::make_tensor<float>(ntt::fixed_shape_v<D0, D1, D2>);
    NttTest::init_tensor(buffer_1, -10.f, 10.f);

    auto ntt_output = ntt::make_tensor<float>(ntt::fixed_shape_v<D0, D1, D2>);

    for (size_t i = 0; i < warmup_num; i++)
        vectorized_softmax(buffer_1, ntt_output, 2_dim, ntt::fixed_shape_v<>);

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        vectorized_softmax(buffer_1, ntt_output, 2_dim, ntt::fixed_shape_v<>);
        asm volatile("" ::"g"(ntt_output));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

void benchmark_ntt_softmax_fixed_reduceAxis1_vectorizeAxis1() {
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
    auto buffer_1 = ntt::make_tensor<float>(ntt::fixed_shape_v<D0, D1, D2>);
    NttTest::init_tensor(buffer_1, -10.f, 10.f);
    auto buffer_2 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<D0, D1 / P, D2>);

    pack(buffer_1, buffer_2, ntt::fixed_shape_v<1>);
    auto buffer_3 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<D0, D1 / P, D2>);

    for (size_t i = 0; i < warmup_num; i++)
        vectorized_softmax(buffer_2, buffer_3, 1_dim, ntt::fixed_shape_v<1>);

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        vectorized_softmax(buffer_2, buffer_3, 1_dim, ntt::fixed_shape_v<1>);
        asm volatile("" ::"g"(buffer_3));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

void benchmark_ntt_softmax_fixed_reduceAxis2_vectorizeAxis2() {
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
    auto buffer_1 = ntt::make_tensor<float>(ntt::fixed_shape_v<D0, D1, D2>);
    NttTest::init_tensor(buffer_1, -10.f, 10.f);
    auto buffer_2 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<D0, D1, D2 / P>);

    pack(buffer_1, buffer_2, ntt::fixed_shape_v<2>);
    auto buffer_3 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<D0, D1, D2 / P>);

    for (size_t i = 0; i < warmup_num; i++)
        vectorized_softmax(buffer_2, buffer_3, 2_dim, ntt::fixed_shape_v<2>);

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        vectorized_softmax(buffer_2, buffer_3, 2_dim, ntt::fixed_shape_v<2>);
        asm volatile("" ::"g"(buffer_3));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

void benchmark_ntt_softmax_fixed_reduceAxis2_vectorizeAxis1() {
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
    auto buffer_1 = ntt::make_tensor<float>(ntt::fixed_shape_v<D0, D1, D2>);
    NttTest::init_tensor(buffer_1, -10.f, 10.f);
    auto buffer_2 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<D0, D1 / P, D2>);

    pack(buffer_1, buffer_2, ntt::fixed_shape_v<1>);
    auto buffer_3 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<D0, D1 / P, D2>);

    for (size_t i = 0; i < warmup_num; i++)
        vectorized_softmax(buffer_2, buffer_3, 2_dim, ntt::fixed_shape_v<1>);

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        vectorized_softmax(buffer_2, buffer_3, 2_dim, ntt::fixed_shape_v<1>);
        asm volatile("" ::"g"(buffer_3));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

void benchmark_ntt_softmax_fixed_reduceAxis1_vectorizeAxis2() {
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
    auto buffer_1 = ntt::make_tensor<float>(ntt::fixed_shape_v<D0, D1, D2>);
    NttTest::init_tensor(buffer_1, -10.f, 10.f);
    auto buffer_2 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<D0, D1, D2 / P>);

    pack(buffer_1, buffer_2, ntt::fixed_shape_v<2>);
    auto buffer_3 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<D0, D1, D2 / P>);

    for (size_t i = 0; i < warmup_num; i++)
        vectorized_softmax(buffer_2, buffer_3, 1_dim, ntt::fixed_shape_v<2>);

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        vectorized_softmax(buffer_2, buffer_3, 1_dim, ntt::fixed_shape_v<2>);
        asm volatile("" ::"g"(buffer_3));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

void benchmark_ntt_softmax_ranked_reduceAxis1_noVectorize() {
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
    auto buffer_1 = ntt::make_tensor<float>(ntt::make_shape(D0, D1, D2));
    NttTest::init_tensor(buffer_1, -10.f, 10.f);

    auto ntt_output = ntt::make_tensor<float>(ntt::make_shape(D0, D1, D2));

    for (size_t i = 0; i < warmup_num; i++)
        vectorized_softmax(buffer_1, ntt_output, 1_dim, ntt::fixed_shape_v<>);

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        vectorized_softmax(buffer_1, ntt_output, 1_dim, ntt::fixed_shape_v<>);
        asm volatile("" ::"g"(ntt_output));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

void benchmark_ntt_softmax_ranked_reduceAxis2_noVectorize() {
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
    auto buffer_1 = ntt::make_tensor<float>(ntt::make_shape(D0, D1, D2));
    NttTest::init_tensor(buffer_1, -10.f, 10.f);

    auto ntt_output = ntt::make_tensor<float>(ntt::make_shape(D0, D1, D2));

    for (size_t i = 0; i < warmup_num; i++)
        vectorized_softmax(buffer_1, ntt_output, 2_dim, ntt::fixed_shape_v<>);

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        vectorized_softmax(buffer_1, ntt_output, 2_dim, ntt::fixed_shape_v<>);
        asm volatile("" ::"g"(ntt_output));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

void benchmark_ntt_softmax_ranked_reduceAxis1_vectorizeAxis1() {
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
    auto buffer_1 = ntt::make_tensor<float>(ntt::make_shape(D0, D1, D2));
    NttTest::init_tensor(buffer_1, -10.f, 10.f);
    auto buffer_2 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::make_shape(D0, D1 / P, D2));

    pack(buffer_1, buffer_2, ntt::fixed_shape_v<1>);
    auto buffer_3 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::make_shape(D0, D1 / P, D2));

    for (size_t i = 0; i < warmup_num; i++)
        vectorized_softmax(buffer_2, buffer_3, 1_dim, ntt::fixed_shape_v<1>);

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        vectorized_softmax(buffer_2, buffer_3, 1_dim, ntt::fixed_shape_v<1>);
        asm volatile("" ::"g"(buffer_3));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

void benchmark_ntt_softmax_ranked_reduceAxis2_vectorizeAxis2() {
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
    auto buffer_1 = ntt::make_tensor<float>(ntt::make_shape(D0, D1, D2));
    NttTest::init_tensor(buffer_1, -10.f, 10.f);
    auto buffer_2 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::make_shape(D0, D1, D2 / P));

    pack(buffer_1, buffer_2, ntt::fixed_shape_v<2>);
    auto buffer_3 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::make_shape(D0, D1, D2 / P));

    for (size_t i = 0; i < warmup_num; i++)
        vectorized_softmax(buffer_2, buffer_3, 2_dim, ntt::fixed_shape_v<2>);

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        vectorized_softmax(buffer_2, buffer_3, 2_dim, ntt::fixed_shape_v<2>);
        asm volatile("" ::"g"(buffer_3));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

void benchmark_ntt_softmax_ranked_reduceAxis2_vectorizeAxis1() {
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
    auto buffer_1 = ntt::make_tensor<float>(ntt::make_shape(D0, D1, D2));
    NttTest::init_tensor(buffer_1, -10.f, 10.f);
    auto buffer_2 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::make_shape(D0, D1 / P, D2));

    pack(buffer_1, buffer_2, ntt::fixed_shape_v<1>);
    auto buffer_3 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::make_shape(D0, D1 / P, D2));

    for (size_t i = 0; i < warmup_num; i++)
        vectorized_softmax(buffer_2, buffer_3, 2_dim, ntt::fixed_shape_v<1>);

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        vectorized_softmax(buffer_2, buffer_3, 2_dim, ntt::fixed_shape_v<1>);
        asm volatile("" ::"g"(buffer_3));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

void benchmark_ntt_softmax_ranked_reduceAxis1_vectorizeAxis2() {
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
    auto buffer_1 = ntt::make_tensor<float>(ntt::make_shape(D0, D1, D2));
    NttTest::init_tensor(buffer_1, -10.f, 10.f);
    auto buffer_2 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::make_shape(D0, D1, D2 / P));

    pack(buffer_1, buffer_2, ntt::fixed_shape_v<2>);
    auto buffer_3 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::make_shape(D0, D1, D2 / P));

    for (size_t i = 0; i < warmup_num; i++)
        vectorized_softmax(buffer_2, buffer_3, 1_dim, ntt::fixed_shape_v<2>);

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        vectorized_softmax(buffer_2, buffer_3, 1_dim, ntt::fixed_shape_v<2>);
        asm volatile("" ::"g"(buffer_3));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

int main() {

    benchmark_ntt_softmax_fixed_reduceAxis1_noVectorize();
    benchmark_ntt_softmax_fixed_reduceAxis2_noVectorize();
    benchmark_ntt_softmax_fixed_reduceAxis1_vectorizeAxis1();
    benchmark_ntt_softmax_fixed_reduceAxis2_vectorizeAxis2();
    benchmark_ntt_softmax_fixed_reduceAxis2_vectorizeAxis1();
    benchmark_ntt_softmax_fixed_reduceAxis1_vectorizeAxis2();

    benchmark_ntt_softmax_ranked_reduceAxis1_noVectorize();
    benchmark_ntt_softmax_ranked_reduceAxis2_noVectorize();
    benchmark_ntt_softmax_ranked_reduceAxis1_vectorizeAxis1();
    benchmark_ntt_softmax_ranked_reduceAxis2_vectorizeAxis2();
    benchmark_ntt_softmax_ranked_reduceAxis2_vectorizeAxis1();
    benchmark_ntt_softmax_ranked_reduceAxis1_vectorizeAxis2();

    return 0;
}