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
#include "nncase/ntt/vector.h"
#include <riscv_vector.h>

namespace nncase::ntt::ukernels {
template <reduce_op Op, class T> struct u_reduce_policy<Op, T, true> {
    static constexpr size_t unroll = 8;
};

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
    static constexpr size_t n0_tile = 4;
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
    static constexpr size_t m0_tile = 4;
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
    static constexpr size_t m0_tile = 4;
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
} // namespace nncase::ntt::ukernels
