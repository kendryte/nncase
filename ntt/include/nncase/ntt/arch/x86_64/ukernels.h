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
#include "nncase/ntt/vector.h"

namespace nncase::ntt::ukernels {

// unary
#define SPECIALIZE_U_UNARY(op, unroll_num)                                     \
    template <typename T>                                                      \
    struct u_unary_policy<ntt::ops::op<vector<T, 8>>, vector<T, 8>, true> {    \
        static constexpr size_t unroll = unroll_num;                           \
    };

SPECIALIZE_U_UNARY(abs, 2)
SPECIALIZE_U_UNARY(ceil, 2)
SPECIALIZE_U_UNARY(floor, 2)
SPECIALIZE_U_UNARY(neg, 2)
SPECIALIZE_U_UNARY(round, 2)
SPECIALIZE_U_UNARY(sign, 2)
SPECIALIZE_U_UNARY(square, 2)

#undef SPECIALIZE_U_UNARY

// binary
#define SPECIALIZE_U_BINARY(op, unroll_num)                                    \
    template <typename T1, typename T2>                                        \
    struct u_binary_policy<ntt::ops::op<vector<T1, 8>, vector<T2, 8>>,         \
                           vector<T1, 8>, vector<T2, 8>, true> {               \
        static constexpr size_t unroll = unroll_num;                           \
    };                                                                         \
                                                                               \
    template <typename T1, typename T2>                                        \
    struct u_binary_policy<ntt::ops::op<T1, vector<T2, 8>>, T1, vector<T2, 8>, \
                           true> {                                             \
        static constexpr size_t unroll = unroll_num;                           \
    };                                                                         \
                                                                               \
    template <typename T1, typename T2>                                        \
    struct u_binary_policy<ntt::ops::op<vector<T1, 8>, T2>, vector<T1, 8>, T2, \
                           true> {                                             \
        static constexpr size_t unroll = unroll_num;                           \
    };

SPECIALIZE_U_BINARY(add, 2)
SPECIALIZE_U_BINARY(sub, 2)
SPECIALIZE_U_BINARY(mul, 2)
SPECIALIZE_U_BINARY(div, 2)
SPECIALIZE_U_BINARY(max, 2)
SPECIALIZE_U_BINARY(min, 2)
SPECIALIZE_U_BINARY(mod, 2)
SPECIALIZE_U_BINARY(floor_mod, 2)

#undef SPECIALIZE_U_BINARY

// pack
template <size_t M, size_t N, size_t MStrides>
class u_pack<M, N, MStrides, true, float, vector<float, 8>> {
  public:
    constexpr void operator()(const float *input,
                              vector<float, 8> *output) noexcept {

        constexpr bool speedup = M == 8 && N % 8 == 0 && MStrides != 1;

        if constexpr (speedup) {

            auto src = reinterpret_cast<const float *>(input);
            auto dst = reinterpret_cast<float *>(output);
            for (size_t j = 0; j < N / M; j++) {

                __m256 row0 = _mm256_loadu_ps(&src[0 * MStrides]);
                __m256 row1 = _mm256_loadu_ps(&src[1 * MStrides]);
                __m256 row2 = _mm256_loadu_ps(&src[2 * MStrides]);
                __m256 row3 = _mm256_loadu_ps(&src[3 * MStrides]);
                __m256 row4 = _mm256_loadu_ps(&src[4 * MStrides]);
                __m256 row5 = _mm256_loadu_ps(&src[5 * MStrides]);
                __m256 row6 = _mm256_loadu_ps(&src[6 * MStrides]);
                __m256 row7 = _mm256_loadu_ps(&src[7 * MStrides]);

                __m256 t0 = _mm256_unpacklo_ps(row0, row1);
                __m256 t1 = _mm256_unpackhi_ps(row0, row1);
                __m256 t2 = _mm256_unpacklo_ps(row2, row3);
                __m256 t3 = _mm256_unpackhi_ps(row2, row3);
                __m256 t4 = _mm256_unpacklo_ps(row4, row5);
                __m256 t5 = _mm256_unpackhi_ps(row4, row5);
                __m256 t6 = _mm256_unpacklo_ps(row6, row7);
                __m256 t7 = _mm256_unpackhi_ps(row6, row7);

                __m256 u0 = _mm256_shuffle_ps(t0, t2, 0x44); // 0x44 -> 01000100
                __m256 u1 = _mm256_shuffle_ps(t0, t2, 0xEE); // 0xEE -> 11101110
                __m256 u2 = _mm256_shuffle_ps(t1, t3, 0x44);
                __m256 u3 = _mm256_shuffle_ps(t1, t3, 0xEE);
                __m256 u4 = _mm256_shuffle_ps(t4, t6, 0x44);
                __m256 u5 = _mm256_shuffle_ps(t4, t6, 0xEE);
                __m256 u6 = _mm256_shuffle_ps(t5, t7, 0x44);
                __m256 u7 = _mm256_shuffle_ps(t5, t7, 0xEE);

                row0 = _mm256_permute2f128_ps(u0, u4, 0x20); // 0x20 -> 00100000
                row1 = _mm256_permute2f128_ps(u1, u5, 0x20);
                row2 = _mm256_permute2f128_ps(u2, u6, 0x20);
                row3 = _mm256_permute2f128_ps(u3, u7, 0x20);
                row4 = _mm256_permute2f128_ps(u0, u4, 0x31); // 0x31 -> 00110001
                row5 = _mm256_permute2f128_ps(u1, u5, 0x31);
                row6 = _mm256_permute2f128_ps(u2, u6, 0x31);
                row7 = _mm256_permute2f128_ps(u3, u7, 0x31);

                _mm256_storeu_ps(&dst[0 * 8], row0);
                _mm256_storeu_ps(&dst[1 * 8], row1);
                _mm256_storeu_ps(&dst[2 * 8], row2);
                _mm256_storeu_ps(&dst[3 * 8], row3);
                _mm256_storeu_ps(&dst[4 * 8], row4);
                _mm256_storeu_ps(&dst[5 * 8], row5);
                _mm256_storeu_ps(&dst[6 * 8], row6);
                _mm256_storeu_ps(&dst[7 * 8], row7);
                src += 8;
                dst += 64;
            }
        } else {
            for (size_t j = 0; j < N; j++) {
                for (size_t i = 0; i < M; i++) {
                    output[j](i) = input[i * MStrides + j];
                }
            }

            if constexpr (M < 8) {
                for (size_t j = 0; j < N; j++) {
                    for (size_t i = M; i < 8; i++) {
                        output[j](i) = 0.f;
                    }
                }
            }
        }
    }
};

template <class TIn, class TOut, size_t... Axes>
class u_pack2d<TIn, TOut, float, vector<float, 8, 8>, Axes...> {
  public:
    constexpr void operator()(const TIn &input, TOut &output) noexcept {
        using TVec = vector<float, 8, 8>;
        constexpr auto axes = std::array<size_t, sizeof...(Axes)>{Axes...};
        constexpr auto in_rank = TIn::rank();
        constexpr auto out_rank = TOut::rank();
        constexpr auto lanes = TVec::shape();
        auto out_shape = output.shape();

        apply(out_shape, [&](auto index) {
            auto out_index = slice_index<out_rank>(index);
            auto in_index = slice_index<in_rank>(index);
            loop<axes.size()>([&](auto i) {
                in_index[axes[i]] = in_index[axes[i]] * lanes[i];
            });
            auto in_ptr =
                reinterpret_cast<const vector<float, 8> *>(&input(in_index));
            auto out_ptr =
                reinterpret_cast<vector<float, 8> *>(&output(out_index));
            for (size_t i = 0; i < lanes[0]; i++) {
                out_ptr[i] = in_ptr[i * out_shape[out_rank - 1]];
            }
        });
    }
};

// reduce
template <reduce_op Op, class T> struct u_reduce_policy<Op, T, true> {
    static constexpr size_t unroll = 8;
};

// gather
template <class T> struct u_memcpy_policy<T, true> {
    static constexpr size_t unroll = 4;
};

// matmul
template <>
struct u_matmul_policy<mamtul_pack_kind::no_pack, float, float, float, true> {
    static constexpr size_t m0_tile = 1;
    static constexpr size_t n0_tile = 1;
    static constexpr size_t m0_subtile = 0;
};

// Pack M
template <>
struct u_matmul_policy<mamtul_pack_kind::pack_m, vector<float, 8>, float,
                       vector<float, 8>, true> {
    static constexpr size_t m0_tile = 2;
    static constexpr size_t n0_tile = 4;
    static constexpr size_t m0_subtile = 0;
};

// Pack K
template <>
struct u_matmul_policy<mamtul_pack_kind::pack_k, vector<float, 8>,
                       vector<float, 8>, float, true> {
    static constexpr size_t m0_tile = 2;
    static constexpr size_t n0_tile = 2;
    static constexpr size_t m0_subtile = 0;
};

// Pack N
template <>
struct u_matmul_policy<mamtul_pack_kind::pack_n, float, vector<float, 8>,
                       vector<float, 8>, true> {
    static constexpr size_t m0_tile = 4;
    static constexpr size_t n0_tile = 2;
    static constexpr size_t m0_subtile = 0;
};

// Pack MN
template <>
struct u_matmul_policy<mamtul_pack_kind::pack_mn, vector<float, 8>,
                       vector<float, 8>, vector<float, 8, 8>, true> {
    static constexpr size_t m0_tile = 1;
    static constexpr size_t n0_tile = 2;
    static constexpr size_t m0_subtile = 4;
};

// Pack MK
template <>
struct u_matmul_policy<mamtul_pack_kind::pack_mk, vector<float, 8, 8>,
                       vector<float, 8>, vector<float, 8>, true> {
    static constexpr size_t m0_tile = 1;
    static constexpr size_t n0_tile = 1;
    static constexpr size_t m0_subtile = 0;
};

// Pack KN
template <>
struct u_matmul_policy<mamtul_pack_kind::pack_kn, vector<float, 8>,
                       vector<float, 8, 8>, vector<float, 8>, true> {
    static constexpr size_t m0_tile = 4;
    static constexpr size_t n0_tile = 2;
    static constexpr size_t m0_subtile = 0;
};

// Pack MKN
template <>
struct u_matmul_policy<mamtul_pack_kind::pack_mkn, vector<float, 8, 8>,
                       vector<float, 8, 8>, vector<float, 8, 8>, true> {
    static constexpr size_t m0_tile = 1;
    static constexpr size_t n0_tile = 2;
    static constexpr size_t m0_subtile = 4;
};
} // namespace nncase::ntt::ukernels
