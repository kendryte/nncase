#include "ntt_test.h"
#include <iomanip>
#include <nncase/ntt/ntt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <type_traits>

using namespace nncase;

// no vectorize
void benchmark_ntt_layernorm_fixed_reduceAxis1_noVectorize() {
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

    auto buffer_0 = ntt::make_tensor<float>(ntt::fixed_shape_v<D0, D1, D2>);
    auto buffer_1 = ntt::make_tensor<float>(ntt::fixed_shape_v<D1, D2>);
    auto buffer_2 = ntt::make_tensor<float>(ntt::fixed_shape_v<D1, D2>);
    std::iota(buffer_0.elements().begin(), buffer_0.elements().end(), 0.f);
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);
    std::iota(buffer_2.elements().begin(), buffer_2.elements().end(), 0.f);

    // no vectorize
    auto ntt_output = ntt::make_tensor<float>(ntt::fixed_shape_v<D0, D1, D2>);

    for (size_t i = 0; i < warmup_num; i++) {
        vectorized_layer_norm(buffer_0, buffer_1, buffer_2, ntt_output, 1e-06,
                          1_dim, ntt::fixed_shape_v<>, ntt::fixed_shape_v<>);
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        vectorized_layer_norm(buffer_0, buffer_1, buffer_2, ntt_output, 1e-06,
                          1_dim, ntt::fixed_shape_v<>, ntt::fixed_shape_v<>);
        asm volatile("" ::"g"(ntt_output));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

void benchmark_ntt_layernorm_fixed_reduceAxis2_noVectorize() {
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

    auto buffer_0 = ntt::make_tensor<float>(ntt::fixed_shape_v<D0, D1, D2>);
    auto buffer_1 = ntt::make_tensor<float>(ntt::fixed_shape_v<D2>);
    auto buffer_2 = ntt::make_tensor<float>(ntt::fixed_shape_v<D2>);
    std::iota(buffer_0.elements().begin(), buffer_0.elements().end(), 0.f);
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 1.f);
    std::iota(buffer_2.elements().begin(), buffer_2.elements().end(), 1.f);

    auto ntt_output = ntt::make_tensor<float>(ntt::fixed_shape_v<D0, D1, D2>);
    // no vectorize
    for (size_t i = 0; i < warmup_num; i++) {
        vectorized_layer_norm(buffer_0, buffer_1, buffer_2, ntt_output, 1e-06,
                          2_dim, ntt::fixed_shape_v<>, ntt::fixed_shape_v<>);
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        vectorized_layer_norm(buffer_0, buffer_1, buffer_2, ntt_output, 1e-06,
                          2_dim, ntt::fixed_shape_v<>, ntt::fixed_shape_v<>);
        asm volatile("" ::"g"(ntt_output));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

void benchmark_ntt_layernorm_fixed_reduceAxis1_vectorizeAxis1() {
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
    auto buffer_0 = ntt::make_tensor<float>(ntt::fixed_shape_v<D0, D1, D2>);
    auto buffer_1 = ntt::make_tensor<float>(ntt::fixed_shape_v<D1, D2>);
    auto buffer_2 = ntt::make_tensor<float>(ntt::fixed_shape_v<D1, D2>);
    std::iota(buffer_0.elements().begin(), buffer_0.elements().end(), 0.f);
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);
    std::iota(buffer_2.elements().begin(), buffer_2.elements().end(), 0.f);

    auto buffer_3 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<D0, D1 / P, D2>);
    auto buffer_4 =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<D1 / P, D2>);
    auto buffer_5 =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<D1 / P, D2>);
    auto buffer_6 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<D0, D1 / P, D2>);
    vectorize(buffer_0, buffer_3, ntt::fixed_shape_v<1>);
    vectorize(buffer_1, buffer_4, ntt::fixed_shape_v<0>);
    vectorize(buffer_2, buffer_5, ntt::fixed_shape_v<0>);

    // no vectorize
    for (size_t i = 0; i < warmup_num; i++) {
        vectorized_layer_norm(buffer_3, buffer_4, buffer_5, buffer_6, 1E-06, 1_dim,
                          ntt::fixed_shape_v<1>, ntt::fixed_shape_v<>);
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        vectorized_layer_norm(buffer_3, buffer_4, buffer_5, buffer_6, 1E-06, 1_dim,
                          ntt::fixed_shape_v<1>, ntt::fixed_shape_v<>);
        asm volatile("" ::"g"(buffer_6));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

void benchmark_ntt_layernorm_fixed_reduceAxis2_vectorizeAxis2() {
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

    auto input = ntt::make_tensor<float>(ntt::fixed_shape_v<D0, D1, D2>);
    auto scale = ntt::make_tensor<float>(ntt::fixed_shape_v<D2>);
    auto bias = ntt::make_tensor<float>(ntt::fixed_shape_v<D2>);
    std::iota(input.elements().begin(), input.elements().end(), 0.f);
    std::iota(scale.elements().begin(), scale.elements().end(), 0.f);
    std::iota(bias.elements().rbegin(), bias.elements().rend(), 0.f);

    auto input_vectorized = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<D0, D1, D2 / P>);
    auto scale_vectorized =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<D2 / P>);
    auto bias_vectorized =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<D2 / P>);
    ntt::vectorize(input, input_vectorized, ntt::fixed_shape_v<2>);
    ntt::vectorize(scale, scale_vectorized, ntt::fixed_shape_v<0>);
    ntt::vectorize(bias, bias_vectorized, ntt::fixed_shape_v<0>);
    auto output_vectorized = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<D0, D1, D2 / P>);

    // no vectorize
    for (size_t i = 0; i < warmup_num; i++) {
        vectorized_layer_norm(input_vectorized, scale_vectorized, bias_vectorized,
                          output_vectorized, 1E-06, 2_dim, ntt::fixed_shape_v<2>,
                          ntt::fixed_shape_v<>);
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        vectorized_layer_norm(input_vectorized, scale_vectorized, bias_vectorized,
                          output_vectorized, 1E-06, 2_dim, ntt::fixed_shape_v<2>,
                          ntt::fixed_shape_v<>);
        asm volatile("" ::"g"(output_vectorized));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

void benchmark_ntt_layernorm_fixed_reduceAxis2_vectorizeAxis1() {
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

    auto buffer_0 = ntt::make_tensor<float>(ntt::fixed_shape_v<D0, D1, D2>);
    auto buffer_1 = ntt::make_tensor<float>(ntt::fixed_shape_v<D2>);
    auto buffer_2 = ntt::make_tensor<float>(ntt::fixed_shape_v<D2>);
    std::iota(buffer_0.elements().begin(), buffer_0.elements().end(), 0.f);
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 1.f);
    std::iota(buffer_2.elements().begin(), buffer_2.elements().end(), 1.f);

    auto buffer_3 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<D0, D1 / P, D2>);
    auto buffer_4 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<D0, D1 / P, D2>);
    vectorize(buffer_0, buffer_3, ntt::fixed_shape_v<1>);

    // no vectorize
    for (size_t i = 0; i < warmup_num; i++) {
        vectorized_layer_norm(buffer_3, buffer_1, buffer_2, buffer_4, 1E-06, 2_dim,
                          ntt::fixed_shape_v<1>, ntt::fixed_shape_v<>);
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        vectorized_layer_norm(buffer_3, buffer_1, buffer_2, buffer_4, 1E-06, 2_dim,
                          ntt::fixed_shape_v<1>, ntt::fixed_shape_v<>);
        asm volatile("" ::"g"(buffer_4));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

void benchmark_ntt_layernorm_fixed_reduceAxis1_vectorizeAxis2() {
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

    auto input = ntt::make_tensor<float>(ntt::fixed_shape_v<D0, D1, D2>);
    auto scale = ntt::make_tensor<float>(ntt::fixed_shape_v<D1, D2>);
    auto bias = ntt::make_tensor<float>(ntt::fixed_shape_v<D1, D2>);
    std::iota(input.elements().begin(), input.elements().end(), 0.f);
    std::iota(scale.elements().begin(), scale.elements().end(), 0.f);
    std::iota(bias.elements().begin(), bias.elements().end(), 0.f);

    // vectorized axis < layer norm axis
    auto vectorized_input = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<D0, D1, D2 / P>);
    auto vectorized_scale =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<D1, D2 / P>);
    auto vectorized_bias =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<D1, D2 / P>);
    auto vectorized_output = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<D0, D1, D2 / P>);
    vectorize(input, vectorized_input, ntt::fixed_shape_v<2>);
    vectorize(scale, vectorized_scale, ntt::fixed_shape_v<1>);
    vectorize(bias, vectorized_bias, ntt::fixed_shape_v<1>);
    // no vectorize
    for (size_t i = 0; i < warmup_num; i++) {
        vectorized_layer_norm(vectorized_input, vectorized_scale, vectorized_bias,
                          vectorized_output, 1E-06, 1_dim, ntt::fixed_shape_v<1>,
                          ntt::fixed_shape_v<>);
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        vectorized_layer_norm(vectorized_input, vectorized_scale, vectorized_bias,
                          vectorized_output, 1E-06, 1_dim, ntt::fixed_shape_v<1>,
                          ntt::fixed_shape_v<>);
        asm volatile("" ::"g"(vectorized_output));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

// no vectorize
void benchmark_ntt_layernorm_ranked_reduceAxis1_noVectorize() {
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

    auto buffer_0 = ntt::make_tensor<float>(ntt::make_shape(D0, D1, D2));
    auto buffer_1 = ntt::make_tensor<float>(ntt::make_shape(D1, D2));
    auto buffer_2 = ntt::make_tensor<float>(ntt::make_shape(D1, D2));
    std::iota(buffer_0.elements().begin(), buffer_0.elements().end(), 0.f);
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);
    std::iota(buffer_2.elements().begin(), buffer_2.elements().end(), 0.f);

    // no vectorize
    auto ntt_output = ntt::make_tensor<float>(ntt::make_shape(D0, D1, D2));

    for (size_t i = 0; i < warmup_num; i++) {
        vectorized_layer_norm(buffer_0, buffer_1, buffer_2, ntt_output, 1e-06,
                          1_dim, ntt::fixed_shape_v<>, ntt::fixed_shape_v<>);
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        vectorized_layer_norm(buffer_0, buffer_1, buffer_2, ntt_output, 1e-06,
                          1_dim, ntt::fixed_shape_v<>, ntt::fixed_shape_v<>);
        asm volatile("" ::"g"(ntt_output));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

void benchmark_ntt_layernorm_ranked_reduceAxis2_noVectorize() {
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

    auto buffer_0 = ntt::make_tensor<float>(ntt::make_shape(D0, D1, D2));
    auto buffer_1 = ntt::make_tensor<float>(ntt::make_shape(D2));
    auto buffer_2 = ntt::make_tensor<float>(ntt::make_shape(D2));
    std::iota(buffer_0.elements().begin(), buffer_0.elements().end(), 0.f);
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 1.f);
    std::iota(buffer_2.elements().begin(), buffer_2.elements().end(), 1.f);

    auto ntt_output = ntt::make_tensor<float>(ntt::make_shape(D0, D1, D2));
    // no vectorize
    for (size_t i = 0; i < warmup_num; i++) {
        vectorized_layer_norm(buffer_0, buffer_1, buffer_2, ntt_output, 1e-06,
                          2_dim, ntt::fixed_shape_v<>, ntt::fixed_shape_v<>);
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        vectorized_layer_norm(buffer_0, buffer_1, buffer_2, ntt_output, 1e-06,
                          2_dim, ntt::fixed_shape_v<>, ntt::fixed_shape_v<>);
        asm volatile("" ::"g"(ntt_output));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

void benchmark_ntt_layernorm_ranked_reduceAxis1_vectorizeAxis1() {
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
    auto buffer_0 = ntt::make_tensor<float>(ntt::make_shape(D0, D1, D2));
    auto buffer_1 = ntt::make_tensor<float>(ntt::make_shape(D1, D2));
    auto buffer_2 = ntt::make_tensor<float>(ntt::make_shape(D1, D2));
    std::iota(buffer_0.elements().begin(), buffer_0.elements().end(), 0.f);
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);
    std::iota(buffer_2.elements().begin(), buffer_2.elements().end(), 0.f);

    auto buffer_3 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::make_shape(D0, D1 / P, D2));
    auto buffer_4 =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(D1 / P, D2));
    auto buffer_5 =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(D1 / P, D2));
    auto buffer_6 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::make_shape(D0, D1 / P, D2));
    vectorize(buffer_0, buffer_3, ntt::fixed_shape_v<1>);
    vectorize(buffer_1, buffer_4, ntt::fixed_shape_v<0>);
    vectorize(buffer_2, buffer_5, ntt::fixed_shape_v<0>);

    // no vectorize
    for (size_t i = 0; i < warmup_num; i++) {
        vectorized_layer_norm(buffer_3, buffer_4, buffer_5, buffer_6, 1E-06, 1_dim,
                          ntt::fixed_shape_v<1>, ntt::fixed_shape_v<>);
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        vectorized_layer_norm(buffer_3, buffer_4, buffer_5, buffer_6, 1E-06, 1_dim,
                          ntt::fixed_shape_v<1>, ntt::fixed_shape_v<>);
        asm volatile("" ::"g"(buffer_6));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

void benchmark_ntt_layernorm_ranked_reduceAxis2_vectorizeAxis2() {
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

    // vectorized axis == layer norm axis
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    auto input = ntt::make_tensor<float>(ntt::make_shape(D0, D1, D2));
    auto scale = ntt::make_tensor<float>(ntt::make_shape(D2));
    auto bias = ntt::make_tensor<float>(ntt::make_shape(D2));
    std::iota(input.elements().begin(), input.elements().end(), 0.f);
    std::iota(scale.elements().begin(), scale.elements().end(), 0.f);
    std::iota(bias.elements().rbegin(), bias.elements().rend(), 0.f);

    auto input_vectorized = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::make_shape(D0, D1, D2 / P));
    auto scale_vectorized =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(D2 / P));
    auto bias_vectorized =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(D2 / P));
    ntt::vectorize(input, input_vectorized, ntt::fixed_shape_v<2>);
    ntt::vectorize(scale, scale_vectorized, ntt::fixed_shape_v<0>);
    ntt::vectorize(bias, bias_vectorized, ntt::fixed_shape_v<0>);
    auto output_vectorized = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<D0, D1, D2 / P>);

    // no vectorize
    for (size_t i = 0; i < warmup_num; i++) {
        vectorized_layer_norm(input_vectorized, scale_vectorized, bias_vectorized,
                          output_vectorized, 1E-06, 2_dim, ntt::fixed_shape_v<2>,
                          ntt::fixed_shape_v<>);
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        vectorized_layer_norm(input_vectorized, scale_vectorized, bias_vectorized,
                          output_vectorized, 1E-06, 2_dim, ntt::fixed_shape_v<2>,
                          ntt::fixed_shape_v<>);
        asm volatile("" ::"g"(output_vectorized));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

void benchmark_ntt_layernorm_ranked_reduceAxis2_vectorizeAxis1() {
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

    // vectorized axis < layer norm axis
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    auto buffer_0 = ntt::make_tensor<float>(ntt::make_shape(D0, D1, D2));
    auto buffer_1 = ntt::make_tensor<float>(ntt::make_shape(D2));
    auto buffer_2 = ntt::make_tensor<float>(ntt::make_shape(D2));
    std::iota(buffer_0.elements().begin(), buffer_0.elements().end(), 0.f);
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 1.f);
    std::iota(buffer_2.elements().begin(), buffer_2.elements().end(), 1.f);

    auto buffer_3 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::make_shape(D0, D1 / P, D2));
    auto buffer_4 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::make_shape(D0, D1 / P, D2));
    vectorize(buffer_0, buffer_3, ntt::fixed_shape_v<1>);

    // no vectorize
    for (size_t i = 0; i < warmup_num; i++) {
        vectorized_layer_norm(buffer_3, buffer_1, buffer_2, buffer_4, 1E-06, 2_dim,
                          ntt::fixed_shape_v<1>, ntt::fixed_shape_v<>);
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        vectorized_layer_norm(buffer_3, buffer_1, buffer_2, buffer_4, 1E-06, 2_dim,
                          ntt::fixed_shape_v<1>, ntt::fixed_shape_v<>);
        asm volatile("" ::"g"(buffer_4));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

void benchmark_ntt_layernorm_ranked_reduceAxis1_vectorizeAxis2() {
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

    auto input = ntt::make_tensor<float>(ntt::make_shape(D0, D1, D2));
    auto scale = ntt::make_tensor<float>(ntt::make_shape(D1, D2));
    auto bias = ntt::make_tensor<float>(ntt::make_shape(D1, D2));
    std::iota(input.elements().begin(), input.elements().end(), 0.f);
    std::iota(scale.elements().begin(), scale.elements().end(), 0.f);
    std::iota(bias.elements().begin(), bias.elements().end(), 0.f);

    // vectorized axis < layer norm axis
    auto vectorized_input = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::make_shape(D0, D1, D2 / P));
    auto vectorized_scale =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(D1, D2 / P));
    auto vectorized_bias =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(D1, D2 / P));
    auto vectorized_output = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::make_shape(D0, D1, D2 / P));
    vectorize(input, vectorized_input, ntt::fixed_shape_v<2>);
    vectorize(scale, vectorized_scale, ntt::fixed_shape_v<1>);
    vectorize(bias, vectorized_bias, ntt::fixed_shape_v<1>);
    // no vectorize
    for (size_t i = 0; i < warmup_num; i++) {
        vectorized_layer_norm(vectorized_input, vectorized_scale, vectorized_bias,
                          vectorized_output, 1E-06, 1_dim, ntt::fixed_shape_v<1>,
                          ntt::fixed_shape_v<>);
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        vectorized_layer_norm(vectorized_input, vectorized_scale, vectorized_bias,
                          vectorized_output, 1E-06, 1_dim, ntt::fixed_shape_v<1>,
                          ntt::fixed_shape_v<>);
        asm volatile("" ::"g"(vectorized_output));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

int main() {

    benchmark_ntt_layernorm_fixed_reduceAxis1_noVectorize();
    benchmark_ntt_layernorm_fixed_reduceAxis2_noVectorize();
    benchmark_ntt_layernorm_fixed_reduceAxis1_vectorizeAxis1();
    benchmark_ntt_layernorm_fixed_reduceAxis2_vectorizeAxis2();
    benchmark_ntt_layernorm_fixed_reduceAxis2_vectorizeAxis1();
    benchmark_ntt_layernorm_fixed_reduceAxis1_vectorizeAxis2();

    benchmark_ntt_layernorm_ranked_reduceAxis1_noVectorize();
    benchmark_ntt_layernorm_ranked_reduceAxis2_noVectorize();
    benchmark_ntt_layernorm_ranked_reduceAxis1_vectorizeAxis1();
    benchmark_ntt_layernorm_ranked_reduceAxis2_vectorizeAxis2();
    benchmark_ntt_layernorm_ranked_reduceAxis2_vectorizeAxis1();
    benchmark_ntt_layernorm_ranked_reduceAxis1_vectorizeAxis2();

    return 0;
}