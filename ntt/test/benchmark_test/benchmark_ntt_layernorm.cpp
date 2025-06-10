#include "ntt_test.h"
#include <iomanip>
#include <nncase/ntt/ntt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <type_traits>

using namespace nncase;

// no pack
void benchmark_ntt_layernorm_fixed_reduceAxis1_noPack() {
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

    ntt::tensor<float, ntt::fixed_shape<D0, D1, D2>> buffer_0;
    ntt::tensor<float, ntt::fixed_shape<D1, D2>> buffer_1;
    ntt::tensor<float, ntt::fixed_shape<D1, D2>> buffer_2;
    std::iota(buffer_0.elements().begin(), buffer_0.elements().end(), 0.f);
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);
    std::iota(buffer_2.elements().begin(), buffer_2.elements().end(), 0.f);

    // no pack
    ntt::tensor<float, ntt::fixed_shape<D0, D1, D2>> ntt_output;

    for (size_t i = 0; i < warmup_num; i++) {
        packed_layer_norm<1>(buffer_0, buffer_1, buffer_2, ntt_output, 1e-06,
                             ntt::fixed_shape<>{}, ntt::fixed_shape<>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        packed_layer_norm<1>(buffer_0, buffer_1, buffer_2, ntt_output, 1e-06,
                             ntt::fixed_shape<>{}, ntt::fixed_shape<>{});
        asm volatile("" ::"g"(ntt_output));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

void benchmark_ntt_layernorm_fixed_reduceAxis2_noPack() {
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

    ntt::tensor<float, ntt::fixed_shape<D0, D1, D2>> buffer_0;
    ntt::tensor<float, ntt::fixed_shape<D2>> buffer_1;
    ntt::tensor<float, ntt::fixed_shape<D2>> buffer_2;
    std::iota(buffer_0.elements().begin(), buffer_0.elements().end(), 0.f);
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 1.f);
    std::iota(buffer_2.elements().begin(), buffer_2.elements().end(), 1.f);

    ntt::tensor<float, ntt::fixed_shape<D0, D1, D2>> ntt_output;
    packed_layer_norm<2>(buffer_0, buffer_1, buffer_2, ntt_output, 1e-06,
                         ntt::fixed_shape<>{}, ntt::fixed_shape<>{});

    // no pack
    for (size_t i = 0; i < warmup_num; i++) {
        packed_layer_norm<2>(buffer_0, buffer_1, buffer_2, ntt_output, 1e-06,
                             ntt::fixed_shape<>{}, ntt::fixed_shape<>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        packed_layer_norm<2>(buffer_0, buffer_1, buffer_2, ntt_output, 1e-06,
                             ntt::fixed_shape<>{}, ntt::fixed_shape<>{});
        asm volatile("" ::"g"(ntt_output));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

void benchmark_ntt_layernorm_fixed_reduceAxis1_packAxis1() {
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
    ntt::tensor<float, ntt::fixed_shape<D0, D1, D2>> buffer_0;
    ntt::tensor<float, ntt::fixed_shape<D1, D2>> buffer_1;
    ntt::tensor<float, ntt::fixed_shape<D1, D2>> buffer_2;
    std::iota(buffer_0.elements().begin(), buffer_0.elements().end(), 0.f);
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);
    std::iota(buffer_2.elements().begin(), buffer_2.elements().end(), 0.f);

    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<D0, D1 / P, D2>>
        buffer_3;
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<D1 / P, D2>> buffer_4;
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<D1 / P, D2>> buffer_5;
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<D0, D1 / P, D2>>
        buffer_6;
    pack<1>(buffer_0, buffer_3);
    pack<0>(buffer_1, buffer_4);
    pack<0>(buffer_2, buffer_5);

    // no pack
    for (size_t i = 0; i < warmup_num; i++) {
        packed_layer_norm<1>(buffer_3, buffer_4, buffer_5, buffer_6,
                             ntt::vector<float, P>::from_scalar(1E-06),
                             ntt::fixed_shape<1>{}, ntt::fixed_shape<0>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        packed_layer_norm<1>(buffer_3, buffer_4, buffer_5, buffer_6,
                             ntt::vector<float, P>::from_scalar(1E-06),
                             ntt::fixed_shape<1>{}, ntt::fixed_shape<0>{});
        asm volatile("" ::"g"(buffer_6));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

void benchmark_ntt_layernorm_fixed_reduceAxis2_packAxis2() {
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

    ntt::tensor<float, ntt::fixed_shape<D0, D1, D2>> input;
    ntt::tensor<float, ntt::fixed_shape<D2>> scale;
    ntt::tensor<float, ntt::fixed_shape<D2>> bias;
    std::iota(input.elements().begin(), input.elements().end(), 0.f);
    std::iota(scale.elements().begin(), scale.elements().end(), 0.f);
    std::iota(bias.elements().rbegin(), bias.elements().rend(), 0.f);

    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<D0, D1, D2 / P>>
        input_packed;
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<D2 / P>> scale_packed;
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<D2 / P>> bias_packed;
    ntt::pack<2>(input, input_packed);
    ntt::pack<0>(scale, scale_packed);
    ntt::pack<0>(bias, bias_packed);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<D0, D1, D2 / P>>
        output_packed;

    // no pack
    for (size_t i = 0; i < warmup_num; i++) {
        packed_layer_norm<2>(input_packed, scale_packed, bias_packed,
                             output_packed,
                             ntt::vector<float, P>::from_scalar(1E-06),
                             ntt::fixed_shape<2>{}, ntt::fixed_shape<0>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        packed_layer_norm<2>(input_packed, scale_packed, bias_packed,
                             output_packed,
                             ntt::vector<float, P>::from_scalar(1E-06),
                             ntt::fixed_shape<2>{}, ntt::fixed_shape<0>{});
        asm volatile("" ::"g"(output_packed));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

void benchmark_ntt_layernorm_fixed_reduceAxis2_packAxis1() {
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

    ntt::tensor<float, ntt::fixed_shape<D0, D1, D2>> buffer_0;
    ntt::tensor<float, ntt::fixed_shape<D2>> buffer_1;
    ntt::tensor<float, ntt::fixed_shape<D2>> buffer_2;
    std::iota(buffer_0.elements().begin(), buffer_0.elements().end(), 0.f);
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 1.f);
    std::iota(buffer_2.elements().begin(), buffer_2.elements().end(), 1.f);

    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<D0, D1 / P, D2>>
        buffer_3;
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<D0, D1 / P, D2>>
        buffer_4;
    pack<1>(buffer_0, buffer_3);

    // no pack
    for (size_t i = 0; i < warmup_num; i++) {
        packed_layer_norm<2>(buffer_3, buffer_1, buffer_2, buffer_4,
                             ntt::vector<float, P>::from_scalar(1E-06),
                             ntt::fixed_shape<1>{}, ntt::fixed_shape<0>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        packed_layer_norm<2>(buffer_3, buffer_1, buffer_2, buffer_4,
                             ntt::vector<float, P>::from_scalar(1E-06),
                             ntt::fixed_shape<1>{}, ntt::fixed_shape<0>{});
        asm volatile("" ::"g"(buffer_4));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

void benchmark_ntt_layernorm_fixed_reduceAxis1_packAxis2() {
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

    ntt::tensor<float, ntt::fixed_shape<D0, D1, D2>> input;
    ntt::tensor<float, ntt::fixed_shape<D1, D2>> scale;
    ntt::tensor<float, ntt::fixed_shape<D1, D2>> bias;
    std::iota(input.elements().begin(), input.elements().end(), 0.f);
    std::iota(scale.elements().begin(), scale.elements().end(), 0.f);
    std::iota(bias.elements().begin(), bias.elements().end(), 0.f);

    // packed axis < layer norm axis
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<D0, D1, D2 / P>>
        packed_input;
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<D1, D2 / P>>
        packed_scale;
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<D1, D2 / P>>
        packed_bias;
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<D0, D1, D2 / P>>
        packed_output;
    pack<2>(input, packed_input);
    pack<1>(scale, packed_scale);
    pack<1>(bias, packed_bias);
    // no pack
    for (size_t i = 0; i < warmup_num; i++) {
        packed_layer_norm<1>(packed_input, packed_scale, packed_bias,
                             packed_output,
                             ntt::vector<float, P>::from_scalar(1E-06),
                             ntt::fixed_shape<1>{}, ntt::fixed_shape<0>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        packed_layer_norm<1>(packed_input, packed_scale, packed_bias,
                             packed_output,
                             ntt::vector<float, P>::from_scalar(1E-06),
                             ntt::fixed_shape<1>{}, ntt::fixed_shape<0>{});
        asm volatile("" ::"g"(packed_output));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

// no pack
void benchmark_ntt_layernorm_ranked_reduceAxis1_noPack() {
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

    using tensor_type1 = ntt::tensor<float, ntt::ranked_shape<3>>;
    using tensor_type2 = ntt::tensor<float, ntt::ranked_shape<2>>;
    auto shape1 = ntt::make_ranked_shape(D0, D1, D2);
    auto shape2 = ntt::make_ranked_shape(D1, D2);

    tensor_type1 buffer_0(shape1);
    tensor_type2 buffer_1(shape2);
    tensor_type2 buffer_2(shape2);
    std::iota(buffer_0.elements().begin(), buffer_0.elements().end(), 0.f);
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);
    std::iota(buffer_2.elements().begin(), buffer_2.elements().end(), 0.f);

    // no pack
    tensor_type1 ntt_output(shape1);

    for (size_t i = 0; i < warmup_num; i++) {
        packed_layer_norm<1>(buffer_0, buffer_1, buffer_2, ntt_output, 1e-06,
                             ntt::fixed_shape<>{}, ntt::fixed_shape<>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        packed_layer_norm<1>(buffer_0, buffer_1, buffer_2, ntt_output, 1e-06,
                             ntt::fixed_shape<>{}, ntt::fixed_shape<>{});
        asm volatile("" ::"g"(ntt_output));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

void benchmark_ntt_layernorm_ranked_reduceAxis2_noPack() {
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

    using tensor_type1 = ntt::tensor<float, ntt::ranked_shape<3>>;
    using tensor_type2 = ntt::tensor<float, ntt::ranked_shape<1>>;
    auto shape1 = ntt::make_ranked_shape(D0, D1, D2);
    auto shape2 = ntt::make_ranked_shape(D2);

    tensor_type1 buffer_0(shape1);
    tensor_type2 buffer_1(shape2);
    tensor_type2 buffer_2(shape2);
    std::iota(buffer_0.elements().begin(), buffer_0.elements().end(), 0.f);
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 1.f);
    std::iota(buffer_2.elements().begin(), buffer_2.elements().end(), 1.f);

    // no pack
    tensor_type1 ntt_output(shape1);
    packed_layer_norm<2>(buffer_0, buffer_1, buffer_2, ntt_output, 1e-06,
                         ntt::fixed_shape<>{}, ntt::fixed_shape<>{});

    // no pack
    for (size_t i = 0; i < warmup_num; i++) {
        packed_layer_norm<2>(buffer_0, buffer_1, buffer_2, ntt_output, 1e-06,
                             ntt::fixed_shape<>{}, ntt::fixed_shape<>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        packed_layer_norm<2>(buffer_0, buffer_1, buffer_2, ntt_output, 1e-06,
                             ntt::fixed_shape<>{}, ntt::fixed_shape<>{});
        asm volatile("" ::"g"(ntt_output));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

void benchmark_ntt_layernorm_ranked_reduceAxis1_packAxis1() {
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
    using tensor_type1 = ntt::tensor<float, ntt::ranked_shape<3>>;
    using tensor_type2 = ntt::tensor<float, ntt::ranked_shape<2>>;
    auto shape1 = ntt::make_ranked_shape(D0, D1, D2);
    auto shape2 = ntt::make_ranked_shape(D1, D2);

    using tensor_type3 =
        ntt::tensor<ntt::vector<float, P>, ntt::ranked_shape<3>>;
    using tensor_type4 =
        ntt::tensor<ntt::vector<float, P>, ntt::ranked_shape<2>>;
    auto shape3 = ntt::make_ranked_shape(D0, D1 / P, D2);
    auto shape4 = ntt::make_ranked_shape(D1 / P, D2);

    tensor_type1 buffer_0(shape1);
    tensor_type2 buffer_1(shape2);
    tensor_type2 buffer_2(shape2);
    std::iota(buffer_0.elements().begin(), buffer_0.elements().end(), 0.f);
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);
    std::iota(buffer_2.elements().begin(), buffer_2.elements().end(), 0.f);

    tensor_type3 buffer_3(shape3);
    tensor_type4 buffer_4(shape4);
    tensor_type4 buffer_5(shape4);
    tensor_type3 buffer_6(shape3);
    pack<1>(buffer_0, buffer_3);
    pack<0>(buffer_1, buffer_4);
    pack<0>(buffer_2, buffer_5);

    // no pack
    for (size_t i = 0; i < warmup_num; i++) {
        packed_layer_norm<1>(buffer_3, buffer_4, buffer_5, buffer_6,
                             ntt::vector<float, P>::from_scalar(1E-06),
                             ntt::fixed_shape<1>{}, ntt::fixed_shape<0>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        packed_layer_norm<1>(buffer_3, buffer_4, buffer_5, buffer_6,
                             ntt::vector<float, P>::from_scalar(1E-06),
                             ntt::fixed_shape<1>{}, ntt::fixed_shape<0>{});
        asm volatile("" ::"g"(buffer_6));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

void benchmark_ntt_layernorm_ranked_reduceAxis2_packAxis2() {
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

    // packed axis == layer norm axis
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    using tensor_type1 = ntt::tensor<float, ntt::ranked_shape<3>>;
    using tensor_type2 = ntt::tensor<float, ntt::ranked_shape<1>>;
    auto shape1 = ntt::make_ranked_shape(D0, D1, D2);
    auto shape2 = ntt::make_ranked_shape(D2);

    using tensor_type3 =
        ntt::tensor<ntt::vector<float, P>, ntt::ranked_shape<3>>;
    using tensor_type4 =
        ntt::tensor<ntt::vector<float, P>, ntt::ranked_shape<1>>;
    auto shape3 = ntt::make_ranked_shape(D0, D1, D2 / P);
    auto shape4 = ntt::make_ranked_shape(D2 / P);

    tensor_type1 input(shape1);
    tensor_type2 scale(shape2);
    tensor_type2 bias(shape2);
    std::iota(input.elements().begin(), input.elements().end(), 0.f);
    std::iota(scale.elements().begin(), scale.elements().end(), 0.f);
    std::iota(bias.elements().rbegin(), bias.elements().rend(), 0.f);

    tensor_type3 input_packed(shape3);
    tensor_type4 scale_packed(shape4);
    tensor_type4 bias_packed(shape4);
    ntt::pack<2>(input, input_packed);
    ntt::pack<0>(scale, scale_packed);
    ntt::pack<0>(bias, bias_packed);
    tensor_type3 output_packed(shape3);

    // no pack
    for (size_t i = 0; i < warmup_num; i++) {
        packed_layer_norm<2>(input_packed, scale_packed, bias_packed,
                             output_packed,
                             ntt::vector<float, P>::from_scalar(1E-06),
                             ntt::fixed_shape<2>{}, ntt::fixed_shape<0>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        packed_layer_norm<2>(input_packed, scale_packed, bias_packed,
                             output_packed,
                             ntt::vector<float, P>::from_scalar(1E-06),
                             ntt::fixed_shape<2>{}, ntt::fixed_shape<0>{});
        asm volatile("" ::"g"(output_packed));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

void benchmark_ntt_layernorm_ranked_reduceAxis2_packAxis1() {
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

    // packed axis < layer norm axis
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    using tensor_type1 = ntt::tensor<float, ntt::ranked_shape<3>>;
    using tensor_type2 = ntt::tensor<float, ntt::ranked_shape<1>>;
    auto shape1 = ntt::make_ranked_shape(D0, D1, D2);
    auto shape2 = ntt::make_ranked_shape(D2);

    using tensor_type3 =
        ntt::tensor<ntt::vector<float, P>, ntt::ranked_shape<3>>;
    using tensor_type4 =
        ntt::tensor<ntt::vector<float, P>, ntt::ranked_shape<1>>;
    auto shape3 = ntt::make_ranked_shape(D0, D1 / P, D2);

    tensor_type1 buffer_0(shape1);
    tensor_type2 buffer_1(shape2);
    tensor_type2 buffer_2(shape2);
    std::iota(buffer_0.elements().begin(), buffer_0.elements().end(), 0.f);
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 1.f);
    std::iota(buffer_2.elements().begin(), buffer_2.elements().end(), 1.f);

    tensor_type3 buffer_3(shape3);
    tensor_type3 buffer_4(shape3);
    pack<1>(buffer_0, buffer_3);
    // no pack
    for (size_t i = 0; i < warmup_num; i++) {
        packed_layer_norm<2>(buffer_3, buffer_1, buffer_2, buffer_4,
                             ntt::vector<float, P>::from_scalar(1E-06),
                             ntt::fixed_shape<1>{}, ntt::fixed_shape<0>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        packed_layer_norm<2>(buffer_3, buffer_1, buffer_2, buffer_4,
                             ntt::vector<float, P>::from_scalar(1E-06),
                             ntt::fixed_shape<1>{}, ntt::fixed_shape<0>{});
        asm volatile("" ::"g"(buffer_4));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

void benchmark_ntt_layernorm_ranked_reduceAxis1_packAxis2() {
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
    using tensor_type1 = ntt::tensor<float, ntt::ranked_shape<3>>;
    using tensor_type2 = ntt::tensor<float, ntt::ranked_shape<2>>;
    auto shape1 = ntt::make_ranked_shape(D0, D1, D2);
    auto shape2 = ntt::make_ranked_shape(D1, D2);

    using tensor_type3 =
        ntt::tensor<ntt::vector<float, P>, ntt::ranked_shape<3>>;
    using tensor_type4 =
        ntt::tensor<ntt::vector<float, P>, ntt::ranked_shape<2>>;
    auto shape3 = ntt::make_ranked_shape(D0, D1, D2 / P);
    auto shape4 = ntt::make_ranked_shape(D1, D2 / P);

    tensor_type1 input(shape1);
    tensor_type2 scale(shape2);
    tensor_type2 bias(shape2);
    std::iota(input.elements().begin(), input.elements().end(), 0.f);
    std::iota(scale.elements().begin(), scale.elements().end(), 0.f);
    std::iota(bias.elements().begin(), bias.elements().end(), 0.f);

    // packed axis < layer norm axis
    tensor_type3 packed_input(shape3);
    tensor_type4 packed_scale(shape4);
    tensor_type4 packed_bias(shape4);
    tensor_type3 packed_output(shape3);
    pack<2>(input, packed_input);
    pack<1>(scale, packed_scale);
    pack<1>(bias, packed_bias);
    // no pack
    for (size_t i = 0; i < warmup_num; i++) {
        packed_layer_norm<1>(packed_input, packed_scale, packed_bias,
                             packed_output,
                             ntt::vector<float, P>::from_scalar(1E-06),
                             ntt::fixed_shape<1>{}, ntt::fixed_shape<0>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        packed_layer_norm<1>(packed_input, packed_scale, packed_bias,
                             packed_output,
                             ntt::vector<float, P>::from_scalar(1E-06),
                             ntt::fixed_shape<1>{}, ntt::fixed_shape<0>{});
        asm volatile("" ::"g"(packed_output));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << " took " << std::setprecision(0) << std::fixed
              << static_cast<float>(t2 - t1) / run_num << " cycles"
              << std::endl;
}

int main() {

    benchmark_ntt_layernorm_fixed_reduceAxis1_noPack();
    benchmark_ntt_layernorm_fixed_reduceAxis2_noPack();
    benchmark_ntt_layernorm_fixed_reduceAxis1_packAxis1();
    benchmark_ntt_layernorm_fixed_reduceAxis2_packAxis2();
    benchmark_ntt_layernorm_fixed_reduceAxis2_packAxis1();
    benchmark_ntt_layernorm_fixed_reduceAxis1_packAxis2();

    benchmark_ntt_layernorm_ranked_reduceAxis1_noPack();
    benchmark_ntt_layernorm_ranked_reduceAxis2_noPack();
    benchmark_ntt_layernorm_ranked_reduceAxis1_packAxis1();
    benchmark_ntt_layernorm_ranked_reduceAxis2_packAxis2();
    benchmark_ntt_layernorm_ranked_reduceAxis2_packAxis1();
    benchmark_ntt_layernorm_ranked_reduceAxis1_packAxis2();

    return 0;
}