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
SPECIALIZE_U_UNARY(floor, 8)
SPECIALIZE_U_UNARY(neg, 8)
SPECIALIZE_U_UNARY(round, 8)
SPECIALIZE_U_UNARY(sign, 8)
SPECIALIZE_U_UNARY(square, 8)

#undef SPECIALIZE_U_UNARY

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

// reduce
template <reduce_op Op, class T> struct u_reduce_policy<Op, T, true> {
    static constexpr size_t unroll = 8;
};

// cast
template <> struct u_cast_policy<true> { static constexpr size_t unroll = 8; };

// matmul
template <>
struct u_matmul_policy<mamtul_pack_kind::no_pack, float, float, float, true> {
    static constexpr size_t m0_tile = 1;
    static constexpr size_t n0_tile = 1;
    static constexpr size_t m0_subtile = 0;
};

// Pack M
template <>
struct u_matmul_policy<mamtul_pack_kind::pack_m, vector<float, NTT_VLEN / 32>,
                       float, vector<float, NTT_VLEN / 32>, true> {
    static constexpr size_t m0_tile = 2;
    static constexpr size_t n0_tile = 8;
    static constexpr size_t m0_subtile = 0;
};

// Pack K
template <>
struct u_matmul_policy<mamtul_pack_kind::pack_k, vector<float, NTT_VLEN / 32>,
                       vector<float, NTT_VLEN / 32>, float, true> {
    static constexpr size_t m0_tile = 2;
    static constexpr size_t n0_tile = 2;
    static constexpr size_t m0_subtile = 0;
};

// Pack N
template <>
struct u_matmul_policy<mamtul_pack_kind::pack_n, float,
                       vector<float, NTT_VLEN / 32>,
                       vector<float, NTT_VLEN / 32>, true> {
    static constexpr size_t m0_tile = 8;
    static constexpr size_t n0_tile = 2;
    static constexpr size_t m0_subtile = 0;
};

// Pack MN
template <>
struct u_matmul_policy<mamtul_pack_kind::pack_mn, vector<float, NTT_VLEN / 32>,
                       vector<float, NTT_VLEN / 32>,
                       vector<float, NTT_VLEN / 32, NTT_VLEN / 32>, true> {
    static constexpr size_t m0_tile = 1;
    static constexpr size_t n0_tile = 2;
    static constexpr size_t m0_subtile = 4;
};

// Pack MK
template <>
struct u_matmul_policy<
    mamtul_pack_kind::pack_mk, vector<float, NTT_VLEN / 32, NTT_VLEN / 32>,
    vector<float, NTT_VLEN / 32>, vector<float, NTT_VLEN / 32>, true> {
    static constexpr size_t m0_tile = 1;
    static constexpr size_t n0_tile = 1;
    static constexpr size_t m0_subtile = 0;
};

// Pack KN
template <>
struct u_matmul_policy<mamtul_pack_kind::pack_kn, vector<float, NTT_VLEN / 32>,
                       vector<float, NTT_VLEN / 32, NTT_VLEN / 32>,
                       vector<float, NTT_VLEN / 32>, true> {
    static constexpr size_t m0_tile = 8;
    static constexpr size_t n0_tile = 2;
    static constexpr size_t m0_subtile = 0;
};

// Pack MKN
template <>
struct u_matmul_policy<mamtul_pack_kind::pack_mkn,
                       vector<float, NTT_VLEN / 32, NTT_VLEN / 32>,
                       vector<float, NTT_VLEN / 32, NTT_VLEN / 32>,
                       vector<float, NTT_VLEN / 32, NTT_VLEN / 32>, true> {
    static constexpr size_t m0_tile = 1;
    static constexpr size_t n0_tile = 2;
    static constexpr size_t m0_subtile = 4;
};

template <bool AccumulateC>
struct u_matmul<ukernels::mamtul_pack_kind::pack_m, AccumulateC, false, false,
                2, 8, vector<float, NTT_VLEN / 32>, float,
                vector<float, NTT_VLEN / 32>, true> {
    template <class TA, class TB, class TC>
    constexpr void operator()(const TA &a, const TB &b, TC &c0,
                              size_t K) noexcept {
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
    }
};
} // namespace nncase::ntt::ukernels
