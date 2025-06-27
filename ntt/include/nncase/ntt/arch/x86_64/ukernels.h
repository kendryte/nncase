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
#include "nncase/ntt/shape.h"
#include "nncase/ntt/vector.h"

namespace nncase::ntt::ukernels {

// unary
#define SPECIALIZE_U_UNARY(op, unroll_num)                                     \
    template <typename T>                                                      \
    struct u_unary_policy<ntt::ops::op<vector<T, 8>>, vector<T, 8>, true> {    \
        static constexpr size_t unroll = unroll_num;                           \
    };

SPECIALIZE_U_UNARY(abs, 2)
SPECIALIZE_U_UNARY(copy, 4)
SPECIALIZE_U_UNARY(ceil, 2)
SPECIALIZE_U_UNARY(floor, 2)
SPECIALIZE_U_UNARY(neg, 2)
SPECIALIZE_U_UNARY(round, 2)
SPECIALIZE_U_UNARY(sign, 2)
SPECIALIZE_U_UNARY(square, 2)

#undef SPECIALIZE_U_UNARY

template <>
struct u_unary<ntt::ops::copy<vector<float, 8>>, vector<float, 8>, true> {
  public:
    void operator()(const ntt::ops::copy<vector<float, 8>> &,
                    const vector<float, 8> *input, size_t input_stride,
                    vector<float, 8> *output, size_t output_stride,
                    size_t count) noexcept {
        using policy_t = u_unary_policy<ntt::ops::copy<vector<float, 8>>,
                                        vector<float, 8>, true>;
        constexpr auto unroll = policy_t::unroll;
        while (count / unroll) {
            for (size_t i = 0; i < unroll; i++) {
                __m256 data =
                    _mm256_loadu_ps(reinterpret_cast<const float *>(input));
                _mm256_storeu_ps(reinterpret_cast<float *>(output), data);
                input += input_stride;
                output += output_stride;
                count--;
            }
        }

        for (size_t i = 0; i < count; i++) {
            __m256 data =
                _mm256_loadu_ps(reinterpret_cast<const float *>(input));
            _mm256_storeu_ps(reinterpret_cast<float *>(output), data);
            input += input_stride;
            output += output_stride;
        }
    }
};

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

// compare
#define SPECIALIZE_U_COMPARE(op, unroll_num)                                   \
    template <typename T1, typename T2>                                        \
    struct u_compare_policy<ntt::ops::op<vector<T1, 8>, vector<T2, 8>>,        \
                            vector<T1, 8>, vector<T2, 8>, true> {              \
        static constexpr size_t unroll = unroll_num;                           \
    };                                                                         \
                                                                               \
    template <typename T1, typename T2>                                        \
    struct u_compare_policy<ntt::ops::op<T1, vector<T2, 8>>, T1,               \
                            vector<T2, 8>, true> {                             \
        static constexpr size_t unroll = unroll_num;                           \
    };                                                                         \
                                                                               \
    template <typename T1, typename T2>                                        \
    struct u_compare_policy<ntt::ops::op<vector<T1, 8>, T2>, vector<T1, 8>,    \
                            T2, true> {                                        \
        static constexpr size_t unroll = unroll_num;                           \
    };

SPECIALIZE_U_COMPARE(equal, 2)
SPECIALIZE_U_COMPARE(not_equal, 2)
SPECIALIZE_U_COMPARE(greater, 2)
SPECIALIZE_U_COMPARE(greater_or_equal, 2)
SPECIALIZE_U_COMPARE(less, 2)
SPECIALIZE_U_COMPARE(less_or_equal, 2)

#undef SPECIALIZE_U_COMPARE

template <typename TElem>
inline void permute_8x8(const TElem *src, TElem *dst, size_t in_stride,
                        size_t out_stride) noexcept {
    __m256 row0 = _mm256_loadu_ps(&src[0 * in_stride]);
    __m256 row1 = _mm256_loadu_ps(&src[1 * in_stride]);
    __m256 row2 = _mm256_loadu_ps(&src[2 * in_stride]);
    __m256 row3 = _mm256_loadu_ps(&src[3 * in_stride]);
    __m256 row4 = _mm256_loadu_ps(&src[4 * in_stride]);
    __m256 row5 = _mm256_loadu_ps(&src[5 * in_stride]);
    __m256 row6 = _mm256_loadu_ps(&src[6 * in_stride]);
    __m256 row7 = _mm256_loadu_ps(&src[7 * in_stride]);

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

    _mm256_storeu_ps(&dst[0 * out_stride], row0);
    _mm256_storeu_ps(&dst[1 * out_stride], row1);
    _mm256_storeu_ps(&dst[2 * out_stride], row2);
    _mm256_storeu_ps(&dst[3 * out_stride], row3);
    _mm256_storeu_ps(&dst[4 * out_stride], row4);
    _mm256_storeu_ps(&dst[5 * out_stride], row5);
    _mm256_storeu_ps(&dst[6 * out_stride], row6);
    _mm256_storeu_ps(&dst[7 * out_stride], row7);
}

// pack
template <> class u_pack<true, float, vector<float, 8>> {
  public:
    template <Dimension TM, Dimension TN, Dimension TMStrides>
    constexpr void operator()(const float *input, const TM &M, const TN &N,
                              const TMStrides &m_strides,
                              vector<float, 8> *output) noexcept {
        constexpr auto speedup_m = 8_dim;
        const bool speedup = M == speedup_m && N % 8 == 0 && m_strides != 1;

        if (speedup) {
            auto src = reinterpret_cast<const float *>(input);
            auto dst = reinterpret_cast<float *>(output);
            for (size_t j = 0; j < N / speedup_m; j++) {
                permute_8x8(src, dst, m_strides, 8);
                src += 8;
                dst += 64;
            }
        } else {
            ukernels::u_pack<false, float, vector<float, 8>> impl;
            impl(input, M, N, m_strides, output);
        }
    }
};

template <FixedTensor TIn, FixedTensor TOut>
class u_pack2d<true, TIn, TOut, float, vector<float, 8, 8>> {
  public:
    template <FixedDimensions TAxes>
    constexpr void operator()(const TIn &input, const TAxes &axes,
                              TOut &output) noexcept {

        constexpr auto axes_temp = TAxes{};

        if constexpr (TAxes::rank() > 0 && (TAxes{}[-1]) == (TIn::rank() - 1)) {
            using TVec = vector<float, 8, 8>;
            constexpr auto in_rank = TIn::rank();
            constexpr auto out_rank = TOut::rank();
            constexpr auto lanes = TVec::shape();
            constexpr auto out_shape = TOut::shape();

            ntt::apply(out_shape, [&](auto index) {
                auto out_index = index;
                auto in_index = index;
                loop<2>([&](auto i) {
                    in_index[axes[i]] = in_index[axes[i]] * lanes[i];
                });
                auto in_ptr = reinterpret_cast<const float *>(&input(in_index));
                auto out_ptr = reinterpret_cast<float *>(&output(out_index));
                for (size_t i = 0; i < lanes[0]; i++) {
                    // out_ptr[i] = in_ptr[i * out_shape[out_rank - 1]];
                    __m256 data = _mm256_loadu_ps(in_ptr);
                    _mm256_storeu_ps(out_ptr, data);
                    in_ptr += lanes[1] * out_shape[out_rank - 1];
                    out_ptr += lanes[1];
                }
            });

        } else {
            using TVec = vector<float, 8, 8>;
            constexpr auto in_rank = TIn::rank();
            constexpr auto out_rank = TOut::rank();
            constexpr auto lanes = TVec::shape();
            const auto out_shape = output.shape();

            const auto domain = out_shape;
            dynamic_shape_t<in_rank> inner_index{};
            dynamic_shape_t<in_rank> outer_index{};

            const auto outer_domain =
                domain.template slice<0, axes_temp[0_dim]>();
            const auto packed_domain =
                domain.template slice<axes_temp[0_dim], axes_temp.rank()>();
            const auto inner_domain =
                domain.template slice<axes_temp[1_dim] + 1>();
            const auto inner_size = inner_domain.length();

            if (inner_size % TVec::shape()[1_dim] != 0) {
                ukernels::u_pack2d<false, TIn, TOut, float, TVec> impl;
                impl(input, axes, output);
            } else {
                ntt::apply(outer_domain, [&](auto index) {
                    loop<axes_temp[0_dim]>([&](auto i) {
                        inner_index[i] = index[i];
                        outer_index[i] = index[i];
                    });
                    for (size_t i = 0; i < packed_domain[0_dim]; i++) {
                        outer_index[axes[0_dim]] = i;
                        auto outer_ptr_keep =
                            reinterpret_cast<float *>(&output(outer_index));
                        for (size_t j = 0; j < lanes[0]; j++) {
                            inner_index[axes[0_dim]] = i * lanes[0_dim] + j;
                            auto outer_ptr = outer_ptr_keep + j * lanes[0];

                            for (size_t k = 0; k < packed_domain[1]; k++) {
                                inner_index[axes[1_dim]] = k * lanes[1];
                                auto input_ptr =
                                    reinterpret_cast<const float *>(
                                        &input(inner_index));

                                for (size_t l = 0;
                                     l < inner_size / lanes[1_dim]; l++) {
                                    auto st_base =
                                        l * lanes[0_dim] * lanes.length();
                                    auto ld_base = l * lanes[1_dim];

                                    auto src = input_ptr + ld_base;
                                    auto dst = outer_ptr + st_base;
                                    permute_8x8(src, dst, inner_size,
                                                lanes.length());
                                }

                                outer_ptr += (inner_size * lanes.length());
                            }
                        }
                    }
                });
            }
        }
    }
};
#if 0
template <size_t axis_stride, class T1, size_t PackAxis>
class u_unpack_1d_fixed<axis_stride, 8, T1, float, true, PackAxis> {
  public:
    void operator()(const T1 &input, size_t input_stride, float *output,
                    size_t count) noexcept {

        constexpr auto in_rank = T1::rank();
        auto in_shape = input.shape();

        dynamic_shape_t<in_rank> inner_domain{};
        dynamic_shape_t<in_rank> domain{};
        for (size_t i = 0; i < in_rank; i++) {
            domain[i] = in_shape[i];
        }

        auto outer_index = slice_index<PackAxis + 1>(domain);
        auto inner_index =
            slice_index<in_rank - (PackAxis + 1)>(domain, PackAxis + 1);
        auto inner_size = inner_index.length();

        auto dst = output;
        if (inner_size % 8 != 0) {
            ukernels::u_unpack_1d_fixed<axis_stride, 8, T1, float, false,
                                        PackAxis>
                impl;
            impl(input, input_stride, output, count);
        } else {
            ntt::apply(outer_index, [&](const auto &index) {
                for (size_t i = 0; i < PackAxis + 1; i++) {
                    inner_domain[i] = index[i];
                }
                auto src =
                    reinterpret_cast<const float *>(&input(inner_domain));

                dst = output + linear_offset(inner_domain, input.strides()) * 8;

                for (size_t i = 0; i < inner_size / 8; i++) {
                    permute_8x8(src, dst, 8, inner_size);
                    dst += 8;
                    src += 64;
                }
            });
        }
    }
};

template <size_t low_axis_stride, size_t high_axis_stride, class TIn,
          size_t Axis1, size_t Axis2>
class u_unpack_2d_fixed<low_axis_stride, 8, high_axis_stride, 8, TIn, float,
                        true, Axis1, Axis2> {
  public:
    void operator()(const TIn &input, size_t input_stride, float *output,
                    size_t count) noexcept {
        using TVec = vector<float, 8, 8>;
        constexpr auto axes = std::array<size_t, 2>{Axis1, Axis2};
        constexpr auto in_rank = TIn::rank();
        auto in_shape = input.shape();

        dynamic_shape_t<in_rank> domain{};
        for (size_t i = 0; i < in_rank; i++) {
            domain[i] = in_shape[i];
        }
        dynamic_shape_t<in_rank> inner_domain{};

        auto packed_index = slice_index<2>(domain, axes[0]);
        auto inner_index =
            slice_index<in_rank - (axes[1] + 1)>(domain, axes[1] + 1);
        auto inner_size = inner_index.length();

        dynamic_shape_t<Axis2> tile_domain{};
        for (size_t i = 0; i < Axis2; i++) {
            tile_domain[i] = in_shape[i];
        }

        auto dst = output;
        if (inner_size % TVec::shape()[1] != 0) {
            ukernels::u_unpack_2d_fixed<low_axis_stride, 8, high_axis_stride, 8,
                                        TIn, float, false, Axis1, Axis2>
                impl;
            impl(input, input_stride, output, count);
        } else {
            ntt::apply(tile_domain, [&](auto index) {
                for (size_t i = 0; i < Axis2; i++) {
                    inner_domain[i] = index[i];
                }
                auto src =
                    reinterpret_cast<const float *>(&input(inner_domain));
                dst =
                    output + linear_offset(inner_domain, input.strides()) * 64;
                for (size_t i = 0; i < 8; i++) {
                    auto st_offset_i = i * packed_index[1] * 8 * inner_size;
                    for (size_t j = 0; j < packed_index[1]; j++) {
                        auto st_offset_j = j * inner_size * 8;
                        auto ld_offset_j = src + j * inner_size * 64;
                        auto st_offset = dst + st_offset_i + st_offset_j;
                        for (size_t k = 0; k < inner_size / 8; k++) {
                            auto st_ptr = st_offset + k * 8;
                            auto ld_ptr = ld_offset_j + k * 512;
                            permute_8x8(ld_ptr, st_ptr, 64, inner_size);
                        }
                    }
                    src = src + 8;
                }
            });
        }
    }
};
#endif

// reduce
template <reduce_op Op, class T> struct u_reduce_policy<Op, T, true> {
    static constexpr size_t unroll = 8;
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

// Where
template <typename T1, typename T2, typename T3>
struct u_where_policy<vector<T1, 8>, vector<T2, 8>, vector<T3, 8>, true> {
    static constexpr size_t unroll = 2;
};

template <typename T1, typename T2, typename T3>
struct u_where_policy<T1, vector<T2, 8>, vector<T3, 8>, true> {
    static constexpr size_t unroll = 2;
};

template <typename T1, typename T2, typename T3>
struct u_where_policy<vector<T1, 8>, T2, vector<T3, 8>, true> {
    static constexpr size_t unroll = 2;
};

template <typename T1, typename T2, typename T3>
struct u_where_policy<vector<T1, 8>, vector<T2, 8>, T3, true> {
    static constexpr size_t unroll = 2;
};
} // namespace nncase::ntt::ukernels
