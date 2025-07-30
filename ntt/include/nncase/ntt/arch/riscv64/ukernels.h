/* Copyright 2019-2021 Canaan Inc.
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
#pragma once
#include "../../ukernels.h"
#include "nncase/ntt/arch/riscv64/arch_types.h"
#include "nncase/ntt/compiler_defs.h"
#include "nncase/ntt/dimension.h"
#include "nncase/ntt/vector.h"
#include <cstddef>
#include <riscv_vector.h>

namespace nncase::ntt::ukernels {

// unary
#define SPECIALIZE_U_UNARY(op, unroll_num)                                     \
    template <typename T>                                                      \
    struct u_unary_policy<ntt::ops::op<vector<T, NTT_VLEN / sizeof(T) / 8>>,   \
                          vector<T, NTT_VLEN / sizeof(T) / 8>, true> {         \
        static constexpr size_t unroll = unroll_num;                           \
    };

SPECIALIZE_U_UNARY(abs, 8)
SPECIALIZE_U_UNARY(ceil, 8)
SPECIALIZE_U_UNARY(copy, 8)
SPECIALIZE_U_UNARY(floor, 8)
SPECIALIZE_U_UNARY(neg, 8)
SPECIALIZE_U_UNARY(round, 8)
SPECIALIZE_U_UNARY(sign, 8)
SPECIALIZE_U_UNARY(square, 8)

#undef SPECIALIZE_U_UNARY

// u_unary<ntt::ops::copy>
template <>
struct u_unary<ntt::ops::copy<vector<float, NTT_VLEN / 32>>,
               vector<float, NTT_VLEN / 32>, true> {
  public:
    void operator()(const ntt::ops::copy<vector<float, NTT_VLEN / 32>> &,
                    const vector<float, NTT_VLEN / 32> *input, size_t in_stride,
                    vector<float, NTT_VLEN / 32> *output, size_t out_stride,
                    size_t count) noexcept {
        using policy_t =
            u_unary_policy<ntt::ops::copy<vector<float, NTT_VLEN / 32>>,
                           vector<float, NTT_VLEN / 32>, true>;
        constexpr auto unroll = policy_t::unroll;
        constexpr auto vl = NTT_VLEN / 32;
        constexpr auto unit = sizeof(vector<float, vl>);
        auto in_strides = in_stride * unit;
        auto out_strides = out_stride * unit;
        asm("vsetvli zero, %[vl], e32, m1, ta, ma\n" ::[vl] "r"(vl));

        while (count / unroll) {
#if 0
              asm volatile(
                  "vl1re32.v v1, (%[input])\n"
                  "add %[input], %[input], %[in_strides]\n"
                  "vl1re32.v v2, (%[input])\n"
                  "add %[input], %[input], %[in_strides]\n"
                  "vs1r.v v1, (%[output])\n"
                  "add %[output], %[output], %[out_strides]\n"
                  "vl1re32.v v3, (%[input])\n"
                  "add %[input], %[input], %[in_strides]\n"
                  "vs1r.v v2, (%[output])\n"
                  "add %[output], %[output], %[out_strides]\n"
                  "vl1re32.v v4, (%[input])\n"
                  "add %[input], %[input], %[in_strides]\n"
                  "vs1r.v v3, (%[output])\n"
                  "add %[output], %[output], %[out_strides]\n"
                  "vs1r.v v4, (%[output])\n"
                  "add %[output], %[output], %[out_strides]\n"
                  : [input] "+r"(input), [output] "+r"(output)
                  : [in_strides] "r"(in_strides), [out_strides] "r"(out_strides));
#else
            asm volatile(
                "vl1re32.v v1, (%[input])\n"
                "add %[input], %[input], %[in_strides]\n"
                "vl1re32.v v2, (%[input])\n"
                "add %[input], %[input], %[in_strides]\n"
                "vs1r.v v1, (%[output])\n"
                "add %[output], %[output], %[out_strides]\n"
                "vl1re32.v v3, (%[input])\n"
                "add %[input], %[input], %[in_strides]\n"
                "vs1r.v v2, (%[output])\n"
                "add %[output], %[output], %[out_strides]\n"
                "vl1re32.v v4, (%[input])\n"
                "add %[input], %[input], %[in_strides]\n"
                "vs1r.v v3, (%[output])\n"
                "add %[output], %[output], %[out_strides]\n"
                "vl1re32.v v5, (%[input])\n"
                "add %[input], %[input], %[in_strides]\n"
                "vs1r.v v4, (%[output])\n"
                "add %[output], %[output], %[out_strides]\n"
                "vl1re32.v v6, (%[input])\n"
                "add %[input], %[input], %[in_strides]\n"
                "vs1r.v v5, (%[output])\n"
                "add %[output], %[output], %[out_strides]\n"
                "vl1re32.v v7, (%[input])\n"
                "add %[input], %[input], %[in_strides]\n"
                "vs1r.v v6, (%[output])\n"
                "add %[output], %[output], %[out_strides]\n"
                "vl1re32.v v8, (%[input])\n"
                "add %[input], %[input], %[in_strides]\n"
                "vs1r.v v7, (%[output])\n"
                "add %[output], %[output], %[out_strides]\n"
                "vs1r.v v8, (%[output])\n"
                "add %[output], %[output], %[out_strides]\n"
                : [input] "+r"(input), [output] "+r"(output)
                : [in_strides] "r"(in_strides), [out_strides] "r"(out_strides));
#endif
            count -= unroll;
        }

        for (size_t i = 0; i < count; i++) {
            *output = *input;
            input += in_stride;
            output += out_stride;
        }
    }
};

// binary
#define SPECIALIZE_U_BINARY(op, unroll_num)                                    \
    template <typename T1, typename T2>                                        \
    struct u_binary_policy<                                                    \
        ntt::ops::op<vector<T1, NTT_VLEN / sizeof(T1) / 8>,                    \
                     vector<T2, NTT_VLEN / sizeof(T2) / 8>>,                   \
        vector<T1, NTT_VLEN / sizeof(T1) / 8>,                                 \
        vector<T2, NTT_VLEN / sizeof(T2) / 8>, true> {                         \
        static constexpr size_t unroll = unroll_num;                           \
    };                                                                         \
                                                                               \
    template <typename T1, typename T2>                                        \
    struct u_binary_policy<                                                    \
        ntt::ops::op<T1, vector<T2, NTT_VLEN / sizeof(T2) / 8>>, T1,           \
        vector<T2, NTT_VLEN / sizeof(T2) / 8>, true> {                         \
        static constexpr size_t unroll = unroll_num;                           \
    };                                                                         \
                                                                               \
    template <typename T1, typename T2>                                        \
    struct u_binary_policy<                                                    \
        ntt::ops::op<vector<T1, NTT_VLEN / sizeof(T1) / 8>, T2>,               \
        vector<T1, NTT_VLEN / sizeof(T1) / 8>, T2, true> {                     \
        static constexpr size_t unroll = unroll_num;                           \
    };

SPECIALIZE_U_BINARY(add, 8)
SPECIALIZE_U_BINARY(sub, 8)
SPECIALIZE_U_BINARY(mul, 8)
SPECIALIZE_U_BINARY(div, 8)
SPECIALIZE_U_BINARY(max, 8)
SPECIALIZE_U_BINARY(min, 8)
SPECIALIZE_U_BINARY(mod, 8)
SPECIALIZE_U_BINARY(floor_mod, 8)

#undef SPECIALIZE_U_BINARY

// clamp
template <> struct u_clamp_policy<true> {
    static constexpr size_t unroll = 8;
};

// reduce
template <reduce_op Op, class T> struct u_reduce_policy<Op, T, true> {
    static constexpr size_t unroll = 8;
};

// cast
template <> struct u_cast_policy<true> {
    static constexpr size_t unroll = 4;
};

// matmul
template <>
struct u_matmul_policy<matmul_pack_kind::no_pack, float, float, float, true> {
    static constexpr size_t m0_tile = 1;
    static constexpr size_t n0_tile = 1;
    static constexpr size_t m0_subtile = 0;
};

// Pack M
template <>
struct u_matmul_policy<matmul_pack_kind::pack_m, vector<float, NTT_VLEN / 32>,
                       float, vector<float, NTT_VLEN / 32>, true> {
    static constexpr size_t m0_tile = 2;
    static constexpr size_t n0_tile = 8;
    static constexpr size_t m0_subtile = 0;
};

// Pack K
template <>
struct u_matmul_policy<matmul_pack_kind::pack_k, vector<float, NTT_VLEN / 32>,
                       vector<float, NTT_VLEN / 32>, float, true> {
    static constexpr size_t m0_tile = 2;
    static constexpr size_t n0_tile = 2;
    static constexpr size_t m0_subtile = 0;
};

// Pack N
template <>
struct u_matmul_policy<matmul_pack_kind::pack_n, float,
                       vector<float, NTT_VLEN / 32>,
                       vector<float, NTT_VLEN / 32>, true> {
    static constexpr size_t m0_tile = 8;
    static constexpr size_t n0_tile = 2;
    static constexpr size_t m0_subtile = 0;
};

// Pack MN
template <>
struct u_matmul_policy<matmul_pack_kind::pack_mn, vector<float, NTT_VLEN / 32>,
                       vector<float, NTT_VLEN / 32>,
                       vector<float, NTT_VLEN / 32, NTT_VLEN / 32>, true> {
    static constexpr size_t m0_tile = 1;
    static constexpr size_t n0_tile = 2;
    static constexpr size_t m0_subtile = 4;
};

// Pack MK
template <>
struct u_matmul_policy<
    matmul_pack_kind::pack_mk, vector<float, NTT_VLEN / 32, NTT_VLEN / 32>,
    vector<float, NTT_VLEN / 32>, vector<float, NTT_VLEN / 32>, true> {
    static constexpr size_t m0_tile = 1;
    static constexpr size_t n0_tile = 1;
    static constexpr size_t m0_subtile = 0;
};

// Pack KN
template <>
struct u_matmul_policy<matmul_pack_kind::pack_kn, vector<float, NTT_VLEN / 32>,
                       vector<float, NTT_VLEN / 32, NTT_VLEN / 32>,
                       vector<float, NTT_VLEN / 32>, true> {
    static constexpr size_t m0_tile = 8;
    static constexpr size_t n0_tile = 2;
    static constexpr size_t m0_subtile = 0;
};

// Pack MKN
template <>
struct u_matmul_policy<matmul_pack_kind::pack_mkn,
                       vector<float, NTT_VLEN / 32, NTT_VLEN / 32>,
                       vector<float, NTT_VLEN / 32, NTT_VLEN / 32>,
                       vector<float, NTT_VLEN / 32, NTT_VLEN / 32>, true> {
    static constexpr size_t m0_tile = 1;
    static constexpr size_t n0_tile = 2;
    static constexpr size_t m0_subtile = 4;
};

template <bool AccumulateC>
struct u_matmul<ukernels::matmul_pack_kind::pack_m, AccumulateC, false, false,
                2, 8, vector<float, NTT_VLEN / 32>, float,
                vector<float, NTT_VLEN / 32>, true> {
    template <class TA, class TB, class TC>
    constexpr void operator()(const TA &a, const TB &b, TC &c0,
                              size_t K) noexcept {
        if constexpr (FixedTensor<TA> && FixedTensor<TB> && FixedTensor<TC>) {
            NTT_ASSUME(K > 0);

            register fixed_vfloat32m1_t c0_0_0 asm("v0") = {};
            register fixed_vfloat32m1_t c0_0_1 asm("v1") = {};
            register fixed_vfloat32m1_t c0_0_2 asm("v2") = {};
            register fixed_vfloat32m1_t c0_0_3 asm("v3") = {};
            register fixed_vfloat32m1_t c0_0_4 asm("v4") = {};
            register fixed_vfloat32m1_t c0_0_5 asm("v5") = {};
            register fixed_vfloat32m1_t c0_0_6 asm("v6") = {};
            register fixed_vfloat32m1_t c0_0_7 asm("v7") = {};
            register fixed_vfloat32m1_t c0_1_0 asm("v8") = {};
            register fixed_vfloat32m1_t c0_1_1 asm("v9") = {};
            register fixed_vfloat32m1_t c0_1_2 asm("v10") = {};
            register fixed_vfloat32m1_t c0_1_3 asm("v11") = {};
            register fixed_vfloat32m1_t c0_1_4 asm("v12") = {};
            register fixed_vfloat32m1_t c0_1_5 asm("v13") = {};
            register fixed_vfloat32m1_t c0_1_6 asm("v14") = {};
            register fixed_vfloat32m1_t c0_1_7 asm("v15") = {};

            if constexpr (AccumulateC) {
                c0_0_0 = c0(0, 0);
                c0_0_1 = c0(0, 1);
                c0_0_2 = c0(0, 2);
                c0_0_3 = c0(0, 3);
                c0_0_4 = c0(0, 4);
                c0_0_5 = c0(0, 5);
                c0_0_6 = c0(0, 6);
                c0_0_7 = c0(0, 7);
                c0_1_0 = c0(1, 0);
                c0_1_1 = c0(1, 1);
                c0_1_2 = c0(1, 2);
                c0_1_3 = c0(1, 3);
                c0_1_4 = c0(1, 4);
                c0_1_5 = c0(1, 5);
                c0_1_6 = c0(1, 6);
                c0_1_7 = c0(1, 7);
            }

            register fixed_vfloat32m1_t a0_0_0 asm("v16");
            register fixed_vfloat32m1_t a0_1_0 asm("v17");
            register fixed_vfloat32m1_t a0_0_1 asm("v18");
            register fixed_vfloat32m1_t a0_1_1 asm("v19");

            register float b0_0_0 asm("fa0");
            register float b0_0_1 asm("fa1");
            register float b0_0_2 asm("fa2");
            register float b0_0_3 asm("fa3");
            register float b0_0_4 asm("fa4");
            register float b0_0_5 asm("fa5");
            register float b0_0_6 asm("fa6");
            register float b0_0_7 asm("fa7");
            register float b0_1_0 asm("ft0");
            register float b0_1_1 asm("ft1");
            register float b0_1_2 asm("ft2");
            register float b0_1_3 asm("ft3");
            register float b0_1_4 asm("ft4");
            register float b0_1_5 asm("ft5");
            register float b0_1_6 asm("ft6");
            register float b0_1_7 asm("ft7");

            const auto a0_strides = a.strides();
            const auto b0_strides = b.strides();
            const auto c0_strides = c0.strides();

            const auto ak_strides = a0_strides[1];
            const auto bk_strides = b0_strides[0];
            const auto bn_strides = b0_strides[1];
            const auto cm_strides = c0_strides[0];
            const auto cn_strides = c0_strides[1];

            {
                register auto a0_0_0_p asm("t0") = a.elements().data();
                register auto a0_1_0_p asm("t1") = a0_0_0_p + a0_strides[0];
                register auto a0_0_1_p asm("t2") = a0_0_0_p + ak_strides;
                register auto a0_1_1_p asm("t3") = a0_0_1_p + a0_strides[0];

                register auto b0_0_x_p asm("t4") = b.elements().data();
                register auto b0_1_x_p asm("t5") = b0_0_x_p + bk_strides;

                // 1. Pre load A/B
                {
                    a0_0_0 = *a0_0_0_p;
                    a0_0_0_p += ak_strides * 2;
                    a0_1_0 = *a0_1_0_p;
                    a0_1_0_p += ak_strides * 2;

                    b0_0_0 = b0_0_x_p[bn_strides * 0];
                    b0_0_1 = b0_0_x_p[bn_strides * 1];
                    b0_0_2 = b0_0_x_p[bn_strides * 2];
                    b0_0_3 = b0_0_x_p[bn_strides * 3];
                    b0_0_4 = b0_0_x_p[bn_strides * 4];
                    b0_0_5 = b0_0_x_p[bn_strides * 5];
                    b0_0_6 = b0_0_x_p[bn_strides * 6];
                    b0_0_7 = b0_0_x_p[bn_strides * 7];
                    b0_0_x_p += bk_strides * 2;
                }

#define NTT_MATMUL_PP(ld, calc)                                                \
    asm volatile(                                                              \
        "vfmacc.vf	%[c0_0_0],%[b0_" #calc "_0],%[a0_0_" #calc "]\n"       \
        "vle32.v %[a0_0_" #ld "], (%[a0_0_" #ld "_p])\n"                       \
        "addi	%[a0_0_" #ld "_p],%[a0_0_" #ld "_p],%[ak_strides] * 2\n"       \
        "vfmacc.vf	%[c0_0_1],%[b0_" #calc "_1],%[a0_0_" #calc "]\n"       \
        "vle32.v %[a0_1_" #ld "], (%[a0_1_" #ld "_p])\n"                       \
        "addi	%[a0_1_" #ld "_p],%[a0_1_" #ld "_p],%[ak_strides] * 2\n"       \
        "vfmacc.vf	%[c0_0_2],%[b0_" #calc "_2],%[a0_0_" #calc "]\n"       \
        "vfmacc.vf	%[c0_0_3],%[b0_" #calc "_3],%[a0_0_" #calc "]\n"       \
        : [a0_0_##ld] "=vr"(a0_0_##ld), [a0_1_##ld] "=vr"(a0_1_##ld),          \
          [c0_0_0] "+vr"(c0_0_0), [c0_0_1] "+vr"(c0_0_1),                      \
          [c0_0_2] "+vr"(c0_0_2), [c0_0_3] "+vr"(c0_0_3),                      \
          [a0_0_##ld##_p] "+r"(a0_0_##ld##_p),                                 \
          [a0_1_##ld##_p] "+r"(a0_1_##ld##_p)                                  \
        : [a0_0_##calc] "vr"(a0_0_##calc), [b0_##calc##_0] "f"(b0_##calc##_0), \
          [b0_##calc##_1] "f"(b0_##calc##_1),                                  \
          [b0_##calc##_2] "f"(b0_##calc##_2),                                  \
          [b0_##calc##_3] "f"(b0_##calc##_3),                                  \
          [ak_strides] "i"(ak_strides * sizeof(a0_0_0)));                      \
                                                                               \
    asm volatile(                                                              \
        "vfmacc.vf	%[c0_0_4],%[b0_" #calc "_4],%[a0_0_" #calc "]\n"       \
        "vfmacc.vf	%[c0_0_5],%[b0_" #calc "_5],%[a0_0_" #calc "]\n"       \
        : [c0_0_4] "+vr"(c0_0_4), [c0_0_5] "+vr"(c0_0_5)                       \
        : [a0_0_##calc] "vr"(a0_0_##calc), [b0_##calc##_4] "f"(b0_##calc##_4), \
          [b0_##calc##_5] "f"(b0_##calc##_5),                                  \
          [ak_strides] "i"(ak_strides * sizeof(a0_0_0)));                      \
                                                                               \
    asm volatile(                                                              \
        "vfmacc.vf	%[c0_0_6],%[b0_" #calc "_6],%[a0_0_" #calc "]\n"       \
        "flw	%[b0_" #ld "_0],0(%[b0_" #ld "_x_p]) \n"                       \
        "vfmacc.vf	%[c0_0_7],%[b0_" #calc "_7],%[a0_0_" #calc "]\n"       \
        "flw	%[b0_" #ld "_1],%[bn_strides](%[b0_" #ld "_x_p]) \n"           \
        : [b0_##ld##_0] "=f"(b0_##ld##_0), [b0_##ld##_1] "=f"(b0_##ld##_1),    \
          [c0_0_6] "+vr"(c0_0_6), [c0_0_7] "+vr"(c0_0_7)                       \
        : [a0_0_##calc] "vr"(a0_0_##calc), [b0_##ld##_x_p] "r"(b0_##ld##_x_p), \
          [b0_##calc##_6] "f"(b0_##calc##_6),                                  \
          [b0_##calc##_7] "f"(b0_##calc##_7),                                  \
          [bn_strides] "i"(bn_strides * sizeof(b0_0_0)));                      \
                                                                               \
    asm volatile(                                                              \
        "vfmacc.vf	%[c0_1_0],%[b0_" #calc "_0],%[a0_1_" #calc "]\n"       \
        "flw	%[b0_" #ld "_2],%[bn_strides] * 2(%[b0_" #ld "_x_p]) \n"       \
        "vfmacc.vf	%[c0_1_1],%[b0_" #calc "_1],%[a0_1_" #calc "]\n"       \
        "flw	%[b0_" #ld "_3],%[bn_strides] * 3(%[b0_" #ld "_x_p]) \n"       \
        "vfmacc.vf	%[c0_1_2],%[b0_" #calc "_2],%[a0_1_" #calc "]\n"       \
        "flw	%[b0_" #ld "_4],%[bn_strides] * 4(%[b0_" #ld "_x_p]) \n"       \
        "vfmacc.vf	%[c0_1_3],%[b0_" #calc "_3],%[a0_1_" #calc "]\n"       \
        "flw	%[b0_" #ld "_5],%[bn_strides] * 5(%[b0_" #ld "_x_p]) \n"       \
        : [b0_##ld##_2] "=f"(b0_##ld##_2), [b0_##ld##_3] "=f"(b0_##ld##_3),    \
          [b0_##ld##_4] "=f"(b0_##ld##_4), [b0_##ld##_5] "=f"(b0_##ld##_5),    \
          [c0_1_0] "+vr"(c0_1_0), [c0_1_1] "+vr"(c0_1_1),                      \
          [c0_1_2] "+vr"(c0_1_2), [c0_1_3] "+vr"(c0_1_3)                       \
        : [a0_1_##calc] "vr"(a0_1_##calc), [b0_##calc##_0] "f"(b0_##calc##_0), \
          [b0_##calc##_1] "f"(b0_##calc##_1),                                  \
          [b0_##calc##_2] "f"(b0_##calc##_2),                                  \
          [b0_##calc##_3] "f"(b0_##calc##_3),                                  \
          [b0_##ld##_x_p] "r"(b0_##ld##_x_p),                                  \
          [bn_strides] "i"(bn_strides * sizeof(b0_0_0)));                      \
                                                                               \
    asm volatile(                                                              \
        "vfmacc.vf	%[c0_1_4],%[b0_" #calc "_4],%[a0_1_" #calc "]\n"       \
        "flw	%[b0_" #ld "_6],%[bn_strides] * 6(%[b0_" #ld "_x_p]) \n"       \
        "vfmacc.vf	%[c0_1_5],%[b0_" #calc "_5],%[a0_1_" #calc "]\n"       \
        "flw	%[b0_" #ld "_7],%[bn_strides] * 7(%[b0_" #ld "_x_p]) \n"       \
        "addi	%[b0_" #ld "_x_p],%[b0_" #ld "_x_p],%[bk_strides] * 2\n"       \
        "vfmacc.vf	%[c0_1_6],%[b0_" #calc "_6],%[a0_1_" #calc "]\n"       \
        "vfmacc.vf	%[c0_1_7],%[b0_" #calc "_7],%[a0_1_" #calc "]\n"       \
        : [b0_##ld##_6] "=f"(b0_##ld##_6), [b0_##ld##_7] "=f"(b0_##ld##_7),    \
          [c0_1_4] "+vr"(c0_1_4), [c0_1_5] "+vr"(c0_1_5),                      \
          [c0_1_6] "+vr"(c0_1_6), [c0_1_7] "+vr"(c0_1_7),                      \
          [b0_##ld##_x_p] "+r"(b0_##ld##_x_p)                                  \
        : [a0_1_##calc] "vr"(a0_1_##calc), [b0_##calc##_4] "f"(b0_##calc##_4), \
          [b0_##calc##_5] "f"(b0_##calc##_5),                                  \
          [b0_##calc##_6] "f"(b0_##calc##_6),                                  \
          [b0_##calc##_7] "f"(b0_##calc##_7),                                  \
          [bk_strides] "i"(bk_strides * sizeof(b0_0_0)),                       \
          [bn_strides] "i"(bn_strides * sizeof(b0_0_0)));

                // 2. Pipelined
                const size_t pipeline_count = (K - 1) / 2;
                for (size_t k1 = 0; k1 < pipeline_count; k1++) {
                    // Ping
                    NTT_MATMUL_PP(1, 0)
                    // Pong
                    NTT_MATMUL_PP(0, 1)
                }

                if (K % 2 == 0) {
                    NTT_MATMUL_PP(1, 0)
                }
            }

            // 3. Tail
            {
                register fixed_vfloat32m1_t *c0_x_0_p asm("t0");
                register fixed_vfloat32m1_t *c0_x_1_p asm("t1");
                register fixed_vfloat32m1_t *c0_x_2_p asm("t2");
                register fixed_vfloat32m1_t *c0_x_3_p asm("t3");
                register fixed_vfloat32m1_t *c0_x_4_p asm("t4");
                register fixed_vfloat32m1_t *c0_x_5_p asm("t5");
                register fixed_vfloat32m1_t *c0_x_6_p asm("t6");
                register fixed_vfloat32m1_t *c0_x_7_p asm("a4");

#define NTT_MATMUL_TAIL(calc)                                                  \
    asm volatile(                                                              \
        "vfmacc.vf	%[c0_0_0],%[b0_" #calc "_0],%[a0_0_" #calc "]\n"       \
        "mv %[c0_x_0_p], %[c0_x_0_p_init]\n"                                   \
        "vfmacc.vf	%[c0_0_1],%[b0_" #calc "_1],%[a0_0_" #calc "]\n"       \
        "addi %[c0_x_1_p], %[c0_x_0_p_init], %[cn_strides]\n"                  \
        "vfmacc.vf	%[c0_0_2],%[b0_" #calc "_2],%[a0_0_" #calc "]\n"       \
        "addi %[c0_x_2_p], %[c0_x_0_p_init], %[cn_strides] * 2\n"              \
        "vfmacc.vf	%[c0_0_3],%[b0_" #calc "_3],%[a0_0_" #calc "]\n"       \
        "addi %[c0_x_3_p], %[c0_x_0_p_init], %[cn_strides] * 3\n"              \
        : [c0_0_0] "+vr"(c0_0_0), [c0_0_1] "+vr"(c0_0_1),                      \
          [c0_0_2] "+vr"(c0_0_2), [c0_0_3] "+vr"(c0_0_3),                      \
          [c0_x_0_p] "=r"(c0_x_0_p), [c0_x_1_p] "=r"(c0_x_1_p),                \
          [c0_x_2_p] "=r"(c0_x_2_p), [c0_x_3_p] "=r"(c0_x_3_p)                 \
        : [a0_0_##calc] "vr"(a0_0_##calc), [b0_##calc##_0] "f"(b0_##calc##_0), \
          [b0_##calc##_1] "f"(b0_##calc##_1),                                  \
          [b0_##calc##_2] "f"(b0_##calc##_2),                                  \
          [b0_##calc##_3] "f"(b0_##calc##_3),                                  \
          [c0_x_0_p_init] "r"(c0.elements().data()),                           \
          [cn_strides] "i"(cn_strides * sizeof(c0_0_0)));                      \
                                                                               \
    asm volatile(                                                              \
        "vfmacc.vf	%[c0_0_4],%[b0_" #calc "_4],%[a0_0_" #calc "]\n"       \
        "vse32.v     %[c0_0_0], (%[c0_x_0_p])\n"                               \
        "add %[c0_x_0_p],%[c0_x_0_p],%[cm_strides] \n"                         \
        "addi %[c0_x_4_p], %[c0_x_0_p_init], %[cn_strides] * 4\n"              \
        "vfmacc.vf	%[c0_0_5],%[b0_" #calc "_5],%[a0_0_" #calc "]\n"       \
        "vse32.v     %[c0_0_1], (%[c0_x_1_p])\n"                               \
        "add %[c0_x_1_p],%[c0_x_1_p],%[cm_strides] \n"                         \
        "addi %[c0_x_5_p], %[c0_x_0_p_init], %[cn_strides] * 5\n"              \
        : [c0_0_4] "+vr"(c0_0_4), [c0_0_5] "+vr"(c0_0_5),                      \
          [c0_x_0_p] "+r"(c0_x_0_p), [c0_x_1_p] "+r"(c0_x_1_p),                \
          [c0_x_4_p] "=r"(c0_x_4_p), [c0_x_5_p] "=r"(c0_x_5_p)                 \
        : [a0_0_##calc] "vr"(a0_0_##calc), [b0_##calc##_4] "f"(b0_##calc##_4), \
          [b0_##calc##_5] "f"(b0_##calc##_5), [c0_0_0] "vr"(c0_0_0),           \
          [c0_0_1] "vr"(c0_0_1), [c0_x_0_p_init] "r"(c0.elements().data()),    \
          [cn_strides] "i"(cn_strides * sizeof(c0_0_0)),                       \
          [cm_strides] "r"(cm_strides * sizeof(c0_0_0)));                      \
                                                                               \
    asm volatile(                                                              \
        "vfmacc.vf	%[c0_0_6],%[b0_" #calc "_6],%[a0_0_" #calc "]\n"       \
        "vse32.v     %[c0_0_2], (%[c0_x_2_p])\n"                               \
        "add %[c0_x_2_p],%[c0_x_2_p],%[cm_strides] \n"                         \
        "addi %[c0_x_6_p], %[c0_x_0_p_init], %[cn_strides] * 6\n"              \
        "vfmacc.vf	%[c0_0_7],%[b0_" #calc "_7],%[a0_0_" #calc "]\n"       \
        "vse32.v     %[c0_0_3], (%[c0_x_3_p])\n"                               \
        "add %[c0_x_3_p],%[c0_x_3_p],%[cm_strides] \n"                         \
        "addi %[c0_x_7_p], %[c0_x_0_p_init], %[cn_strides] * 7\n"              \
        : [c0_0_6] "+vr"(c0_0_6), [c0_0_7] "+vr"(c0_0_7),                      \
          [c0_x_2_p] "+r"(c0_x_2_p), [c0_x_3_p] "+r"(c0_x_3_p),                \
          [c0_x_6_p] "=r"(c0_x_6_p), [c0_x_7_p] "=r"(c0_x_7_p)                 \
        : [a0_0_##calc] "vr"(a0_0_##calc), [b0_##calc##_6] "f"(b0_##calc##_6), \
          [b0_##calc##_7] "f"(b0_##calc##_7), [c0_0_2] "vr"(c0_0_2),           \
          [c0_0_3] "vr"(c0_0_3), [c0_x_0_p_init] "r"(c0.elements().data()),    \
          [cn_strides] "i"(cn_strides * sizeof(c0_0_0)),                       \
          [cm_strides] "r"(cm_strides * sizeof(c0_0_0)));                      \
                                                                               \
    asm volatile(                                                              \
        "vfmacc.vf	%[c0_1_0],%[b0_" #calc "_0],%[a0_1_" #calc "]\n"       \
        "vse32.v     %[c0_0_4], (%[c0_x_4_p])\n"                               \
        "add %[c0_x_4_p],%[c0_x_4_p],%[cm_strides] \n"                         \
        "vfmacc.vf	%[c0_1_1],%[b0_" #calc "_1],%[a0_1_" #calc "]\n"       \
        "vse32.v     %[c0_0_5], (%[c0_x_5_p])\n"                               \
        "add %[c0_x_5_p],%[c0_x_5_p],%[cm_strides] \n"                         \
        "vfmacc.vf	%[c0_1_2],%[b0_" #calc "_2],%[a0_1_" #calc "]\n"       \
        "vse32.v     %[c0_0_6], (%[c0_x_6_p])\n"                               \
        "add %[c0_x_6_p],%[c0_x_6_p],%[cm_strides] \n"                         \
        "vfmacc.vf	%[c0_1_3],%[b0_" #calc "_3],%[a0_1_" #calc "]\n"       \
        "vse32.v     %[c0_0_7], (%[c0_x_7_p])\n"                               \
        "add %[c0_x_7_p],%[c0_x_7_p],%[cm_strides] \n"                         \
        : [c0_1_0] "+vr"(c0_1_0), [c0_1_1] "+vr"(c0_1_1),                      \
          [c0_1_2] "+vr"(c0_1_2), [c0_1_3] "+vr"(c0_1_3),                      \
          [c0_x_4_p] "+r"(c0_x_4_p), [c0_x_5_p] "+r"(c0_x_5_p),                \
          [c0_x_6_p] "+r"(c0_x_6_p), [c0_x_7_p] "+r"(c0_x_7_p)                 \
        : [a0_1_##calc] "vr"(a0_1_##calc), [b0_##calc##_0] "f"(b0_##calc##_0), \
          [b0_##calc##_1] "f"(b0_##calc##_1),                                  \
          [b0_##calc##_2] "f"(b0_##calc##_2),                                  \
          [b0_##calc##_3] "f"(b0_##calc##_3), [c0_0_4] "vr"(c0_0_4),           \
          [c0_0_5] "vr"(c0_0_5), [c0_0_6] "vr"(c0_0_6), [c0_0_7] "vr"(c0_0_7), \
          [cm_strides] "r"(cm_strides * sizeof(c0_0_0)));                      \
                                                                               \
    asm volatile(                                                              \
        "vfmacc.vf	%[c0_1_4],%[b0_" #calc "_4],%[a0_1_" #calc "]\n"       \
        "vse32.v     %[c0_1_0], (%[c0_x_0_p])\n"                               \
        "vfmacc.vf	%[c0_1_5],%[b0_" #calc "_5],%[a0_1_" #calc "]\n"       \
        "vse32.v     %[c0_1_1], (%[c0_x_1_p])\n"                               \
        "vfmacc.vf	%[c0_1_6],%[b0_" #calc "_6],%[a0_1_" #calc "]\n"       \
        "vse32.v     %[c0_1_2], (%[c0_x_2_p])\n"                               \
        "vfmacc.vf	%[c0_1_7],%[b0_" #calc "_7],%[a0_1_" #calc "]\n"       \
        "vse32.v     %[c0_1_3], (%[c0_x_3_p])\n"                               \
        : [c0_1_4] "+vr"(c0_1_4), [c0_1_5] "+vr"(c0_1_5),                      \
          [c0_1_6] "+vr"(c0_1_6), [c0_1_7] "+vr"(c0_1_7)                       \
        : [a0_1_##calc] "vr"(a0_1_##calc), [b0_##calc##_4] "f"(b0_##calc##_4), \
          [b0_##calc##_5] "f"(b0_##calc##_5),                                  \
          [b0_##calc##_6] "f"(b0_##calc##_6),                                  \
          [b0_##calc##_7] "f"(b0_##calc##_7), [c0_1_0] "vr"(c0_1_0),           \
          [c0_1_1] "vr"(c0_1_1), [c0_1_2] "vr"(c0_1_2), [c0_1_3] "vr"(c0_1_3), \
          [c0_x_0_p] "r"(c0_x_0_p), [c0_x_1_p] "r"(c0_x_1_p),                  \
          [c0_x_2_p] "r"(c0_x_2_p), [c0_x_3_p] "r"(c0_x_3_p));                 \
                                                                               \
    asm volatile(                                                              \
        "vse32.v     %[c0_1_4], (%[c0_x_4_p])\n"                               \
        "vse32.v     %[c0_1_5], (%[c0_x_5_p])\n"                               \
        "vse32.v     %[c0_1_6], (%[c0_x_6_p])\n"                               \
        "vse32.v     %[c0_1_7], (%[c0_x_7_p])\n" ::[c0_1_4] "vr"(c0_1_4),      \
        [c0_1_5] "vr"(c0_1_5), [c0_1_6] "vr"(c0_1_6), [c0_1_7] "vr"(c0_1_7),   \
        [c0_x_4_p] "r"(c0_x_4_p), [c0_x_5_p] "r"(c0_x_5_p),                    \
        [c0_x_6_p] "r"(c0_x_6_p), [c0_x_7_p] "r"(c0_x_7_p));

                if (K % 2 == 0) {
                    NTT_MATMUL_TAIL(1)
                } else {
                    NTT_MATMUL_TAIL(0)
                }
            }

#undef NTT_MATMUL_PING
#undef NTT_MATMUL_TAIL
        } else {
            u_matmul<ukernels::matmul_pack_kind::pack_m, AccumulateC, false,
                     false, 2, 8, vector<float, NTT_VLEN / 32>, float,
                     vector<float, NTT_VLEN / 32>, false>
                impl;
            impl(a, b, c0, K);
        }
    }
};

// pack
template <class T1, class T2> struct u_pack_policy<T1, T2, true> {
    static constexpr size_t unroll = 4;
};

template <> class u_pack<true, float, vector<float, NTT_VLEN / 32>> {
  public:
    template <Dimension TM, Dimension TN, Dimension TMStrides>
    constexpr void operator()(const float *input, const TM &M, const TN &N,
                              const TMStrides &m_strides,
                              vector<float, NTT_VLEN / 32> *output) noexcept {
        constexpr size_t vl = NTT_VLEN / 32;
        if (N % vl != 0) {
            ukernels::u_pack<false, float, vector<float, vl>> impl;
            impl(input, M, N, m_strides, output);
        } else {
            using policy_t = u_pack_policy<float, vector<float, vl>, true>;
            constexpr auto unroll = policy_t::unroll;
            const auto in_strides1 = sizeof(float) * m_strides;
            const auto in_strides2 = sizeof(float);
            asm("vsetvli zero, %[vl], e32, m1, ta, ma\n" ::[vl] "r"(vl));

            size_t count = N;
            while (count / unroll) {
                asm volatile(
                    "vlse32.v v1, (%[input]), %[in_strides1]\n"
                    "add %[input], %[input], %[in_strides2]\n"
                    : [input] "+r"(input)
                    : [in_strides1] "r"(in_strides1), [in_strides2] "r"(
                                                          in_strides2));
                auto output1 = output;

                asm volatile(
                    "vlse32.v v2, (%[input]), %[in_strides1]\n"
                    "add %[input], %[input], %[in_strides2]\n"
                    : [input] "+r"(input)
                    : [in_strides1] "r"(in_strides1), [in_strides2] "r"(
                                                          in_strides2));
                auto output2 = output + 1;

                asm volatile(
                    "vlse32.v v3, (%[input]), %[in_strides1]\n"
                    "add %[input], %[input], %[in_strides2]\n"
                    : [input] "+r"(input)
                    : [in_strides1] "r"(in_strides1), [in_strides2] "r"(
                                                          in_strides2));
                auto output3 = output + 2;

                asm volatile(
                    "vlse32.v v4, (%[input]), %[in_strides1]\n"
                    "add %[input], %[input], %[in_strides2]\n"
                    : [input] "+r"(input)
                    : [in_strides1] "r"(in_strides1), [in_strides2] "r"(
                                                          in_strides2));
                auto output4 = output + 3;

                asm volatile("vse32.v v1, (%[output1])\n"
                             : [output1] "+r"(output1)
                             :);
                output += unroll;

                asm volatile("vse32.v v2, (%[output2])\n"
                             : [output2] "+r"(output2)
                             :);
                count -= unroll;

                asm volatile("vse32.v v3, (%[output3])\n"
                             : [output3] "+r"(output3)
                             :);

                asm volatile("vse32.v v4, (%[output4])\n"
                             : [output4] "+r"(output4)
                             :);
            }

            for (size_t i = 0; i < count; i++) {
                asm volatile(
                    "vlse32.v v1, (%[input]), %[in_strides1]\n"
                    "add %[input], %[input], %[in_strides2]\n"
                    "vse32.v v1, (%[output])\n"
                    : [input] "+r"(input), [output] "+r"(output)
                    : [in_strides1] "r"(in_strides1), [in_strides2] "r"(
                                                          in_strides2));
                output += 1;
            }
        }
    }
};

template <class TIn, class TOut>
class u_pack2d<true, TIn, TOut, float,
               vector<float, NTT_VLEN / 32, NTT_VLEN / 32>> {
  public:
    template <FixedDimensions TAxes>
    constexpr void operator()(const TIn &input, const TAxes &,
                              TOut &output) noexcept {
        constexpr auto PackAxis1 = TAxes{}[0_dim];
        constexpr auto PackAxis2 = TAxes{}[1_dim];
        constexpr size_t vl = NTT_VLEN / 32;
        auto input_shape = input.shape();
        auto out_stride = output.strides();
        auto rank = input_shape.rank();
        if ((input_shape[PackAxis1] % vl == 0) &&
            (input_shape[PackAxis2] % vl == 0)) {
            auto pin = input.buffer().data();
            auto out_ptr = output.buffer().data();
            using policy_t =
                u_pack_policy<vector<float, NTT_VLEN / 32, NTT_VLEN / 32>,
                              float, true>;
            constexpr auto unroll1 = 2;
            constexpr auto unroll2 = policy_t::unroll;
            size_t out_offset = 0;
            size_t low_idx = 0;
            size_t high_idx = 0;
            auto low_stride = out_stride[PackAxis1];
            auto high_stride = out_stride[PackAxis2];
            auto high_dim = low_stride / high_stride;
            auto in_low_strides = low_stride * vl * sizeof(float);
            auto in_high_strides = high_stride * sizeof(float);
            auto low_extra = low_stride * (vl * vl - 1);
            auto high_extra = high_stride * (vl - 1);
            auto out_strides = sizeof(vector<float, vl>);
            asm("vsetvli zero, %[vl], e32, m1, ta, ma\n" ::[vl] "r"(vl));

            size_t count = output.shape().length();
            if (PackAxis2 != rank - 1) {
                while (count / high_stride) {
                    auto in_ptr = pin + out_offset + low_idx * low_extra +
                                  high_idx * high_extra;
                    size_t count1 = high_stride;
                    while (count1 / unroll1) {
                        auto input1 = in_ptr;
                        auto input2 = in_ptr + 1;
                        auto output1 = out_ptr;
                        auto output2 = out_ptr + 1;
                        size_t count2 = vl;
                        while (count2 / unroll2) {
                            // load input1 + input2
                            asm volatile(
                                "vlse32.v v1, (%[input1]), %[in_high_strides]\n"
                                "add %[input1], %[input1], %[in_low_strides]\n"
                                : [input1] "+r"(input1)
                                : [in_high_strides] "r"(in_high_strides),
                                  [in_low_strides] "r"(in_low_strides));

                            asm volatile(
                                "vlse32.v v5, (%[input2]), %[in_high_strides]\n"
                                "add %[input2], %[input2], %[in_low_strides]\n"
                                : [input2] "+r"(input2)
                                : [in_high_strides] "r"(in_high_strides),
                                  [in_low_strides] "r"(in_low_strides));

                            asm volatile(
                                "vlse32.v v2, (%[input1]), %[in_high_strides]\n"
                                "add %[input1], %[input1], %[in_low_strides]\n"
                                : [input1] "+r"(input1)
                                : [in_high_strides] "r"(in_high_strides),
                                  [in_low_strides] "r"(in_low_strides));

                            asm volatile(
                                "vlse32.v v6, (%[input2]), %[in_high_strides]\n"
                                "add %[input2], %[input2], %[in_low_strides]\n"
                                : [input2] "+r"(input2)
                                : [in_high_strides] "r"(in_high_strides),
                                  [in_low_strides] "r"(in_low_strides));

                            asm volatile(
                                "vlse32.v v3, (%[input1]), %[in_high_strides]\n"
                                "add %[input1], %[input1], %[in_low_strides]\n"
                                : [input1] "+r"(input1)
                                : [in_high_strides] "r"(in_high_strides),
                                  [in_low_strides] "r"(in_low_strides));

                            asm volatile(
                                "vlse32.v v7, (%[input2]), %[in_high_strides]\n"
                                "add %[input2], %[input2], %[in_low_strides]\n"
                                : [input2] "+r"(input2)
                                : [in_high_strides] "r"(in_high_strides),
                                  [in_low_strides] "r"(in_low_strides));

                            asm volatile(
                                "vlse32.v v4, (%[input1]), %[in_high_strides]\n"
                                "add %[input1], %[input1], %[in_low_strides]\n"
                                : [input1] "+r"(input1)
                                : [in_high_strides] "r"(in_high_strides),
                                  [in_low_strides] "r"(in_low_strides));

                            asm volatile(
                                "vlse32.v v8, (%[input2]), %[in_high_strides]\n"
                                "add %[input2], %[input2], %[in_low_strides]\n"
                                : [input2] "+r"(input2)
                                : [in_high_strides] "r"(in_high_strides),
                                  [in_low_strides] "r"(in_low_strides));

                            // store output1 + output2
                            asm volatile(
                                "vse32.v v1, (%[output1])\n"
                                "add %[output1], %[output1], %[out_strides]\n"
                                : [output1] "+r"(output1)
                                : [out_strides] "r"(out_strides));
                            count2 -= unroll2;

                            asm volatile(
                                "vse32.v v5, (%[output2])\n"
                                "add %[output2], %[output2], %[out_strides]\n"
                                : [output2] "+r"(output2)
                                : [out_strides] "r"(out_strides));

                            asm volatile(
                                "vse32.v v2, (%[output1])\n"
                                "add %[output1], %[output1], %[out_strides]\n"
                                : [output1] "+r"(output1)
                                : [out_strides] "r"(out_strides));

                            asm volatile(
                                "vse32.v v6, (%[output2])\n"
                                "add %[output2], %[output2], %[out_strides]\n"
                                : [output2] "+r"(output2)
                                : [out_strides] "r"(out_strides));

                            asm volatile(
                                "vse32.v v3, (%[output1])\n"
                                "add %[output1], %[output1], %[out_strides]\n"
                                : [output1] "+r"(output1)
                                : [out_strides] "r"(out_strides));

                            asm volatile(
                                "vse32.v v7, (%[output2])\n"
                                "add %[output2], %[output2], %[out_strides]\n"
                                : [output2] "+r"(output2)
                                : [out_strides] "r"(out_strides));

                            asm volatile(
                                "vse32.v v4, (%[output1])\n"
                                "add %[output1], %[output1], %[out_strides]\n"
                                : [output1] "+r"(output1)
                                : [out_strides] "r"(out_strides));

                            asm volatile(
                                "vse32.v v8, (%[output2])\n"
                                "add %[output2], %[output2], %[out_strides]\n"
                                : [output2] "+r"(output2)
                                : [out_strides] "r"(out_strides));
                        }

                        in_ptr += unroll1;
                        out_ptr += unroll1;
                        count1 -= unroll1;
                    }

                    // count1 left
                    for (size_t i = 0; i < count1; i++) {
                        for (size_t j = 0; j < vl; j++) {
                            asm volatile(
                                "vlse32.v v1, (%[in_ptr]), %[in_high_strides]\n"
                                "add %[in_ptr], %[in_ptr], %[in_low_strides]\n"
                                : [in_ptr] "+r"(in_ptr)
                                : [in_high_strides] "r"(in_high_strides),
                                  [in_low_strides] "r"(in_low_strides));

                            asm volatile(
                                "vse32.v v1, (%[out_ptr])\n"
                                "add %[out_ptr], %[out_ptr], %[out_strides]\n"
                                : [out_ptr] "+r"(out_ptr)
                                : [out_strides] "r"(out_strides));
                        }
                    }

                    out_offset += high_stride;
                    high_idx++;
                    low_idx += high_idx / high_dim;
                    high_idx %= high_dim;
                    count -= high_stride;
                }
            } else {
                while (count / high_dim) {
                    auto in_ptr = pin + out_offset + low_idx * low_extra;
                    size_t count1 = high_dim;
                    while (count1 / unroll1) {
                        auto input1 = in_ptr;
                        auto input2 = in_ptr + vl;
                        auto output1 = out_ptr;
                        auto output2 = out_ptr + 1;
                        size_t count2 = vl;
                        while (count2 / unroll2) {
                            // load input1 + input2
                            asm volatile(
                                "vl1re32.v v1, (%[input1])\n"
                                "add %[input1], %[input1], %[in_low_strides]\n"
                                : [input1] "+r"(input1)
                                : [in_low_strides] "r"(in_low_strides));

                            asm volatile(
                                "vl1re32.v v5, (%[input2])\n"
                                "add %[input2], %[input2], %[in_low_strides]\n"
                                : [input2] "+r"(input2)
                                : [in_low_strides] "r"(in_low_strides));

                            asm volatile(
                                "vl1re32.v v2, (%[input1])\n"
                                "add %[input1], %[input1], %[in_low_strides]\n"
                                : [input1] "+r"(input1)
                                : [in_low_strides] "r"(in_low_strides));

                            asm volatile(
                                "vl1re32.v v6, (%[input2])\n"
                                "add %[input2], %[input2], %[in_low_strides]\n"
                                : [input2] "+r"(input2)
                                : [in_low_strides] "r"(in_low_strides));

                            asm volatile(
                                "vl1re32.v v3, (%[input1])\n"
                                "add %[input1], %[input1], %[in_low_strides]\n"
                                : [input1] "+r"(input1)
                                : [in_low_strides] "r"(in_low_strides));

                            asm volatile(
                                "vl1re32.v v7, (%[input2])\n"
                                "add %[input2], %[input2], %[in_low_strides]\n"
                                : [input2] "+r"(input2)
                                : [in_low_strides] "r"(in_low_strides));

                            asm volatile(
                                "vl1re32.v v4, (%[input1])\n"
                                "add %[input1], %[input1], %[in_low_strides]\n"
                                : [input1] "+r"(input1)
                                : [in_low_strides] "r"(in_low_strides));

                            asm volatile(
                                "vl1re32.v v8, (%[input2])\n"
                                "add %[input2], %[input2], %[in_low_strides]\n"
                                : [input2] "+r"(input2)
                                : [in_low_strides] "r"(in_low_strides));

                            // store output1 + output2
                            asm volatile(
                                "vs1r.v v1, (%[output1])\n"
                                "add %[output1], %[output1], %[out_strides]\n"
                                : [output1] "+r"(output1)
                                : [out_strides] "r"(out_strides));
                            count2 -= unroll2;

                            asm volatile(
                                "vs1r.v v5, (%[output2])\n"
                                "add %[output2], %[output2], %[out_strides]\n"
                                : [output2] "+r"(output2)
                                : [out_strides] "r"(out_strides));

                            asm volatile(
                                "vs1r.v v2, (%[output1])\n"
                                "add %[output1], %[output1], %[out_strides]\n"
                                : [output1] "+r"(output1)
                                : [out_strides] "r"(out_strides));

                            asm volatile(
                                "vs1r.v v6, (%[output2])\n"
                                "add %[output2], %[output2], %[out_strides]\n"
                                : [output2] "+r"(output2)
                                : [out_strides] "r"(out_strides));

                            asm volatile(
                                "vs1r.v v3, (%[output1])\n"
                                "add %[output1], %[output1], %[out_strides]\n"
                                : [output1] "+r"(output1)
                                : [out_strides] "r"(out_strides));

                            asm volatile(
                                "vs1r.v v7, (%[output2])\n"
                                "add %[output2], %[output2], %[out_strides]\n"
                                : [output2] "+r"(output2)
                                : [out_strides] "r"(out_strides));

                            asm volatile(
                                "vs1r.v v4, (%[output1])\n"
                                "add %[output1], %[output1], %[out_strides]\n"
                                : [output1] "+r"(output1)
                                : [out_strides] "r"(out_strides));

                            asm volatile(
                                "vs1r.v v8, (%[output2])\n"
                                "add %[output2], %[output2], %[out_strides]\n"
                                : [output2] "+r"(output2)
                                : [out_strides] "r"(out_strides));
                        }

                        in_ptr += unroll1 * vl;
                        out_ptr += unroll1;
                        count1 -= unroll1;
                    }

                    // count1 left
                    for (size_t i = 0; i < count1; i++) {
                        for (size_t j = 0; j < vl; j++) {
                            asm volatile(
                                "vl1re32.v v1, (%[in_ptr])\n"
                                "add %[in_ptr], %[in_ptr], %[in_low_strides]\n"
                                "vs1r.v v1, (%[out_ptr])\n"
                                "add %[out_ptr], %[out_ptr], %[out_strides]\n"
                                : [in_ptr] "+r"(in_ptr), [out_ptr] "+r"(out_ptr)
                                : [in_low_strides] "r"(in_low_strides),
                                  [out_strides] "r"(out_strides));
                        }
                    }

                    low_idx++;
                    out_offset += high_dim;
                    count -= high_dim;
                }
            }
        } else {
            ukernels::u_pack2d<false, TIn, TOut, float, vector<float, vl, vl>>
                impl;
            impl(input, TAxes{}, output);
        }
    }
};

template <class T1, class T2> struct u_unpack_policy<T1, T2, true> {
    static constexpr size_t unroll = 4;
};

template <Tensor TIn, Tensor TOut, size_t AxesRank>
    requires((std::same_as<typename TIn::element_type,
                           ntt::vector<float, NTT_VLEN / 32, NTT_VLEN / 32>> ||
              std::same_as<typename TIn::element_type,
                           ntt::vector<float, NTT_VLEN / 32>>) &&
             std::same_as<typename std::decay_t<TOut>::element_type, float> &&
             (AxesRank == 1 || AxesRank == 2))
class u_unpack_impl<TIn, TOut, AxesRank, true> {
  public:
    using TVec = typename TIn::element_type;
    using TElem = typename std::decay_t<TOut>::element_type;

    template <FixedDimensions TAxes>
    constexpr void operator()(const TIn &input, TOut &output,
                              [[maybe_unused]] const TAxes &axes) {

        auto in_shape = input.shape();
        constexpr auto const_axes = TAxes{};
        constexpr auto in_rank = TIn::rank();
        constexpr auto axis = const_axes[0];
        dynamic_shape_t<in_rank> domain;
        ntt::loop<in_rank>([&](auto &i) { domain[i] = in_shape[i]; });
        auto inner_index =
            domain.template slice<axis + 1, in_rank - (axis + 1)>();
        auto inner_size = inner_index.length();
        constexpr auto vector_size = NTT_VLEN / 32;
        auto axis_stride = input.strides()[const_axes[0]];

        if constexpr (AxesRank == 1) {
            if constexpr (const_axes[0] == (TIn::rank() - 1)) {
                auto size = output.size() * sizeof(TElem);
                auto in_ptr = input.buffer().data();
                auto out_ptr = output.buffer().data();
                std::memcpy(out_ptr, in_ptr, size);
            } else if (inner_size % vector_size == 0 && axis_stride !=0 ) {

                auto in_stride = 1;
                auto count = input.size();
                auto output_local_ptr = output.buffer().data();

                auto in_ptr = input.buffer().data();
                constexpr size_t vl = NTT_VLEN / 32;
                using policy_t =
                    u_unpack_policy<vector<float, vl>, float, true>;
                constexpr auto unroll = policy_t::unroll;
                auto in_strides = in_stride * sizeof(vector<float, vl>);
                auto out_strides = axis_stride * sizeof(float);
                asm("vsetvli zero, %[vl], e32, m1, ta, ma\n" ::[vl] "r"(vl));
                size_t in_offset = 0;
                size_t axis_idx = 0;
                size_t extra = (vl - 1) * axis_stride;
                while (count / axis_stride) {
                    auto tmp = axis_stride;
                    while (tmp / unroll) {
                        auto out_ptr = output_local_ptr + in_offset;
                        asm volatile("vl1re32.v v1, (%[in_ptr])\n"
                                     "add %[in_ptr], %[in_ptr], %[in_strides]\n"
                                     : [in_ptr] "+r"(in_ptr)
                                     : [in_strides] "r"(in_strides));
                        auto output1 = out_ptr + 0 + axis_idx * extra;

                        asm volatile("vl1re32.v v2, (%[in_ptr])\n"
                                     "add %[in_ptr], %[in_ptr], %[in_strides]\n"
                                     : [in_ptr] "+r"(in_ptr)
                                     : [in_strides] "r"(in_strides));
                        auto output2 = out_ptr + 1 + axis_idx * extra;

                        asm volatile("vl1re32.v v3, (%[in_ptr])\n"
                                     "add %[in_ptr], %[in_ptr], %[in_strides]\n"
                                     : [in_ptr] "+r"(in_ptr)
                                     : [in_strides] "r"(in_strides));
                        auto output3 = out_ptr + 2 + axis_idx * extra;

                        asm volatile("vl1re32.v v4, (%[in_ptr])\n"
                                     "add %[in_ptr], %[in_ptr], %[in_strides]\n"
                                     : [in_ptr] "+r"(in_ptr)
                                     : [in_strides] "r"(in_strides));
                        auto output4 = out_ptr + 3 + axis_idx * extra;

                        asm volatile(
                            "vsse32.v v1, (%[output1]), %[out_strides]\n"
                            : [output1] "+r"(output1)
                            : [out_strides] "r"(out_strides));
                        in_offset += unroll;

                        asm volatile(
                            "vsse32.v v2, (%[output2]), %[out_strides]\n"
                            : [output2] "+r"(output2)
                            : [out_strides] "r"(out_strides));
                        count -= unroll;

                        asm volatile(
                            "vsse32.v v3, (%[output3]), %[out_strides]\n"
                            : [output3] "+r"(output3)
                            : [out_strides] "r"(out_strides));
                        tmp -= unroll;

                        asm volatile(
                            "vsse32.v v4, (%[output4]), %[out_strides]\n"
                            : [output4] "+r"(output4)
                            : [out_strides] "r"(out_strides));
                    }

                    for (size_t i = 0; i < tmp; i++) {
                        auto output1 =
                            output_local_ptr + in_offset + axis_idx * extra;
                        asm volatile(
                            "vl1re32.v v1, (%[in_ptr])\n"
                            "add %[in_ptr], %[in_ptr], %[in_strides]\n"
                            "vsse32.v v1, (%[output1]), %[out_strides]\n"
                            : [in_ptr] "+r"(in_ptr), [output1] "+r"(output1)
                            : [in_strides] "r"(in_strides), [out_strides] "r"(
                                                                out_strides));
                        in_offset++;
                        count--;
                    }
                    axis_idx++;
                }
            } else {
                ukernels::u_unpack_impl<TIn, std::decay_t<TOut>, TAxes::rank(),
                                        false>
                    impl;
                impl(input, output, axes);
            }
        } else if (AxesRank == 2 && const_axes[1] == const_axes[0] + 1 &&
                   (inner_size % vector_size == 0)) {
            auto in_stride = 1;
            auto low_stride = input.strides()[const_axes[0]];
            auto high_stride = input.strides()[const_axes[1]];
            auto output_local_ptr = output.buffer().data();
            [[maybe_unused]] auto PackAxis1 = const_axes[0];
            auto PackAxis2 = const_axes[1];
            auto count = input.size();
            auto in_ptr = input.buffer().data();
            constexpr size_t vl = NTT_VLEN / 32;
            using policy_t =
                u_unpack_policy<vector<float, NTT_VLEN / 32, NTT_VLEN / 32>,
                                float, true>;
            constexpr auto unroll1 = 2;
            constexpr auto unroll2 = policy_t::unroll;
            size_t in_offset = 0;
            size_t low_idx = 0;
            size_t high_idx = 0;
            auto high_dim = low_stride / high_stride;
            auto low_extra = low_stride * (vl * vl - 1);
            auto high_extra = high_stride * (vl - 1);
            auto in_strides = sizeof(vector<float, vl>);
            auto out_low_strides = low_stride * vl * sizeof(float);
            auto out_high_strides = high_stride * sizeof(float);
            asm("vsetvli zero, %[vl], e32, m1, ta, ma\n" ::[vl] "r"(vl));

            auto rank = input.shape().rank();
            if (PackAxis2 != rank - 1) {
                while (count / high_stride) {
                    auto out_ptr = output_local_ptr + in_offset +
                                   low_idx * low_extra + high_idx * high_extra;
                    size_t count1 = high_stride;
                    while (count1 / unroll1) {
                        auto input1 = in_ptr;
                        auto input2 = in_ptr + in_stride;
                        auto output1 = out_ptr;
                        auto output2 = out_ptr + 1;
                        size_t count2 = vl;
                        while (count2 / unroll2) {
                            // load input1 + input2
                            asm volatile(
                                "vl1re32.v v1, (%[input1])\n"
                                "add %[input1], %[input1], %[in_strides]\n"
                                : [input1] "+r"(input1)
                                : [in_strides] "r"(in_strides));

                            asm volatile(
                                "vl1re32.v v5, (%[input2])\n"
                                "add %[input2], %[input2], %[in_strides]\n"
                                : [input2] "+r"(input2)
                                : [in_strides] "r"(in_strides));

                            asm volatile(
                                "vl1re32.v v2, (%[input1])\n"
                                "add %[input1], %[input1], %[in_strides]\n"
                                : [input1] "+r"(input1)
                                : [in_strides] "r"(in_strides));

                            asm volatile(
                                "vl1re32.v v6, (%[input2])\n"
                                "add %[input2], %[input2], %[in_strides]\n"
                                : [input2] "+r"(input2)
                                : [in_strides] "r"(in_strides));

                            asm volatile(
                                "vl1re32.v v3, (%[input1])\n"
                                "add %[input1], %[input1], %[in_strides]\n"
                                : [input1] "+r"(input1)
                                : [in_strides] "r"(in_strides));

                            asm volatile(
                                "vl1re32.v v7, (%[input2])\n"
                                "add %[input2], %[input2], %[in_strides]\n"
                                : [input2] "+r"(input2)
                                : [in_strides] "r"(in_strides));

                            asm volatile(
                                "vl1re32.v v4, (%[input1])\n"
                                "add %[input1], %[input1], %[in_strides]\n"
                                : [input1] "+r"(input1)
                                : [in_strides] "r"(in_strides));

                            asm volatile(
                                "vl1re32.v v8, (%[input2])\n"
                                "add %[input2], %[input2], %[in_strides]\n"
                                : [input2] "+r"(input2)
                                : [in_strides] "r"(in_strides));

                            // store output1 + output2
                            asm volatile(
                                "vsse32.v v1, (%[output1]), "
                                "%[out_high_strides]\n"
                                "add %[output1], %[output1], "
                                "%[out_low_strides]\n"
                                : [output1] "+r"(output1)
                                : [out_high_strides] "r"(out_high_strides),
                                  [out_low_strides] "r"(out_low_strides));
                            count2 -= unroll2;

                            asm volatile(
                                "vsse32.v v5, (%[output2]), "
                                "%[out_high_strides]\n"
                                "add %[output2], %[output2], "
                                "%[out_low_strides]\n"
                                : [output2] "+r"(output2)
                                : [out_high_strides] "r"(out_high_strides),
                                  [out_low_strides] "r"(out_low_strides));

                            asm volatile(
                                "vsse32.v v2, (%[output1]), "
                                "%[out_high_strides]\n"
                                "add %[output1], %[output1], "
                                "%[out_low_strides]\n"
                                : [output1] "+r"(output1)
                                : [out_high_strides] "r"(out_high_strides),
                                  [out_low_strides] "r"(out_low_strides));

                            asm volatile(
                                "vsse32.v v6, (%[output2]), "
                                "%[out_high_strides]\n"
                                "add %[output2], %[output2], "
                                "%[out_low_strides]\n"
                                : [output2] "+r"(output2)
                                : [out_high_strides] "r"(out_high_strides),
                                  [out_low_strides] "r"(out_low_strides));

                            asm volatile(
                                "vsse32.v v3, (%[output1]), "
                                "%[out_high_strides]\n"
                                "add %[output1], %[output1], "
                                "%[out_low_strides]\n"
                                : [output1] "+r"(output1)
                                : [out_high_strides] "r"(out_high_strides),
                                  [out_low_strides] "r"(out_low_strides));

                            asm volatile(
                                "vsse32.v v7, (%[output2]), "
                                "%[out_high_strides]\n"
                                "add %[output2], %[output2], "
                                "%[out_low_strides]\n"
                                : [output2] "+r"(output2)
                                : [out_high_strides] "r"(out_high_strides),
                                  [out_low_strides] "r"(out_low_strides));

                            asm volatile(
                                "vsse32.v v4, (%[output1]), "
                                "%[out_high_strides]\n"
                                "add %[output1], %[output1], "
                                "%[out_low_strides]\n"
                                : [output1] "+r"(output1)
                                : [out_high_strides] "r"(out_high_strides),
                                  [out_low_strides] "r"(out_low_strides));

                            asm volatile(
                                "vsse32.v v8, (%[output2]), "
                                "%[out_high_strides]\n"
                                "add %[output2], %[output2], "
                                "%[out_low_strides]\n"
                                : [output2] "+r"(output2)
                                : [out_high_strides] "r"(out_high_strides),
                                  [out_low_strides] "r"(out_low_strides));
                        }

                        // count2 left
                        while (count2 % unroll2) {
                            asm volatile(
                                "vl1re32.v v1, (%[input1])\n"
                                "add %[input1], %[input1], %[in_strides]\n"
                                "vl1re32.v v2, (%[input2])\n"
                                "add %[input2], %[input2], %[in_strides]\n"
                                "vsse32.v v1, (%[output1]), "
                                "%[out_high_strides]\n"
                                "add %[output1], %[output1], "
                                "%[out_low_strides]\n"
                                "vsse32.v v2, (%[output2]), "
                                "%[out_high_strides]\n"
                                "add %[output2], %[output2], "
                                "%[out_low_strides]\n"
                                : [input1] "+r"(input1), [input2] "+r"(input2),
                                  [output1] "+r"(output1),
                                  [output2] "+r"(output2)
                                : [in_strides] "r"(in_strides),
                                  [out_high_strides] "r"(out_high_strides),
                                  [out_low_strides] "r"(out_low_strides));
                            count2--;
                        }

                        in_ptr += in_stride * unroll1;
                        out_ptr += unroll1;
                        count1 -= unroll1;
                    }

                    // count1 left
                    for (size_t i = 0; i < count1; i++) {
                        for (size_t j = 0; j < vl; j++) {
                            asm volatile(
                                "vl1re32.v v1, (%[in_ptr])\n"
                                "add %[in_ptr], %[in_ptr], %[in_strides]\n"
                                "vsse32.v v1, (%[out_ptr]), "
                                "%[out_high_strides]\n"
                                "add %[out_ptr], %[out_ptr], "
                                "%[out_low_strides]\n"
                                : [in_ptr] "+r"(in_ptr), [out_ptr] "+r"(out_ptr)
                                : [in_strides] "r"(in_strides),
                                  [out_high_strides] "r"(out_high_strides),
                                  [out_low_strides] "r"(out_low_strides));
                        }
                    }

                    in_offset += high_stride;
                    high_idx++;
                    low_idx += high_idx / high_dim;
                    high_idx %= high_dim;
                    count -= high_stride;
                }
            } else {
                while (count / high_dim) {
                    auto out_ptr =
                        output_local_ptr + in_offset + low_idx * low_extra;
                    size_t count1 = high_dim;
                    while (count1 / unroll1) {
                        auto input1 = in_ptr;
                        auto input2 = in_ptr + in_stride;
                        auto output1 = out_ptr;
                        auto output2 = out_ptr + vl;
                        size_t count2 = vl;
                        while (count2 / unroll2) {
                            // load input1 + input2
                            asm volatile(
                                "vl1re32.v v1, (%[input1])\n"
                                "add %[input1], %[input1], %[in_strides]\n"
                                : [input1] "+r"(input1)
                                : [in_strides] "r"(in_strides));

                            asm volatile(
                                "vl1re32.v v5, (%[input2])\n"
                                "add %[input2], %[input2], %[in_strides]\n"
                                : [input2] "+r"(input2)
                                : [in_strides] "r"(in_strides));

                            asm volatile(
                                "vl1re32.v v2, (%[input1])\n"
                                "add %[input1], %[input1], %[in_strides]\n"
                                : [input1] "+r"(input1)
                                : [in_strides] "r"(in_strides));

                            asm volatile(
                                "vl1re32.v v6, (%[input2])\n"
                                "add %[input2], %[input2], %[in_strides]\n"
                                : [input2] "+r"(input2)
                                : [in_strides] "r"(in_strides));

                            asm volatile(
                                "vl1re32.v v3, (%[input1])\n"
                                "add %[input1], %[input1], %[in_strides]\n"
                                : [input1] "+r"(input1)
                                : [in_strides] "r"(in_strides));

                            asm volatile(
                                "vl1re32.v v7, (%[input2])\n"
                                "add %[input2], %[input2], %[in_strides]\n"
                                : [input2] "+r"(input2)
                                : [in_strides] "r"(in_strides));

                            asm volatile(
                                "vl1re32.v v4, (%[input1])\n"
                                "add %[input1], %[input1], %[in_strides]\n"
                                : [input1] "+r"(input1)
                                : [in_strides] "r"(in_strides));

                            asm volatile(
                                "vl1re32.v v8, (%[input2])\n"
                                "add %[input2], %[input2], %[in_strides]\n"
                                : [input2] "+r"(input2)
                                : [in_strides] "r"(in_strides));

                            // store output1 + output2
                            asm volatile(
                                "vs1r.v v1, (%[output1])\n"
                                "add %[output1], %[output1], "
                                "%[out_low_strides]\n"
                                : [output1] "+r"(output1)
                                : [out_low_strides] "r"(out_low_strides));
                            count2 -= unroll2;

                            asm volatile(
                                "vs1r.v v5, (%[output2])\n"
                                "add %[output2], %[output2], "
                                "%[out_low_strides]\n"
                                : [output2] "+r"(output2)
                                : [out_low_strides] "r"(out_low_strides));

                            asm volatile(
                                "vs1r.v v2, (%[output1])\n"
                                "add %[output1], %[output1], "
                                "%[out_low_strides]\n"
                                : [output1] "+r"(output1)
                                : [out_low_strides] "r"(out_low_strides));

                            asm volatile(
                                "vs1r.v v6, (%[output2])\n"
                                "add %[output2], %[output2], "
                                "%[out_low_strides]\n"
                                : [output2] "+r"(output2)
                                : [out_low_strides] "r"(out_low_strides));

                            asm volatile(
                                "vs1r.v v3, (%[output1])\n"
                                "add %[output1], %[output1], "
                                "%[out_low_strides]\n"
                                : [output1] "+r"(output1)
                                : [out_low_strides] "r"(out_low_strides));

                            asm volatile(
                                "vs1r.v v7, (%[output2])\n"
                                "add %[output2], %[output2], "
                                "%[out_low_strides]\n"
                                : [output2] "+r"(output2)
                                : [out_low_strides] "r"(out_low_strides));

                            asm volatile(
                                "vs1r.v v4, (%[output1])\n"
                                "add %[output1], %[output1], "
                                "%[out_low_strides]\n"
                                : [output1] "+r"(output1)
                                : [out_low_strides] "r"(out_low_strides));

                            asm volatile(
                                "vs1r.v v8, (%[output2])\n"
                                "add %[output2], %[output2], "
                                "%[out_low_strides]\n"
                                : [output2] "+r"(output2)
                                : [out_low_strides] "r"(out_low_strides));
                        }

                        // count2 left
                        while (count2 % unroll2) {
                            asm volatile(
                                "vl1re32.v v1, (%[input1])\n"
                                "add %[input1], %[input1], %[in_strides]\n"
                                "vl1re32.v v2, (%[input2])\n"
                                "add %[input2], %[input2], %[in_strides]\n"
                                "vs1r.v v1, (%[output1])\n"
                                "add %[output1], %[output1], "
                                "%[out_low_strides]\n"
                                "vs1r.v v2, (%[output2])\n"
                                "add %[output2], %[output2], "
                                "%[out_low_strides]\n"
                                : [input1] "+r"(input1), [input2] "+r"(input2),
                                  [output1] "+r"(output1),
                                  [output2] "+r"(output2)
                                : [in_strides] "r"(in_strides),
                                  [out_low_strides] "r"(out_low_strides));
                            count2--;
                        }

                        in_ptr += in_stride * unroll1;
                        out_ptr += unroll1 * vl;
                        count1 -= unroll1;
                    }

                    // count1 left
                    for (size_t i = 0; i < count1; i++) {
                        for (size_t j = 0; j < vl; j++) {
                            asm volatile(
                                "vl1re32.v v1, (%[in_ptr])\n"
                                "add %[in_ptr], %[in_ptr], %[in_strides]\n"
                                "vs1r.v v1, (%[out_ptr])\n"
                                "add %[out_ptr], %[out_ptr], "
                                "%[out_low_strides]\n"
                                : [in_ptr] "+r"(in_ptr), [out_ptr] "+r"(out_ptr)
                                : [in_strides] "r"(in_strides),
                                  [out_low_strides] "r"(out_low_strides));
                        }
                    }

                    in_offset += high_dim;
                    low_idx += 1;
                    count -= high_dim;
                }
            }
        } else {
            ukernels::u_unpack_impl<TIn, std::decay_t<TOut>, TAxes::rank(),
                                    false>
                impl;
            impl(input, output, axes);
        }
    }
};

} // namespace nncase::ntt::ukernels
