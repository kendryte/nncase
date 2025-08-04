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
#include "../../vector.h"
#include <immintrin.h>

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
                *output = *input;
                input += input_stride;
                output += output_stride;
                count--;
            }
        }

        for (size_t i = 0; i < count; i++) {
            *output = *input;
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

inline void permute_8x8_devectorize1d(const vector<float, 8> *src, float *dst,
                                 size_t in_stride, size_t out_stride) noexcept {
    __m256 row0 = src[0 * in_stride];
    __m256 row1 = src[1 * in_stride];
    __m256 row2 = src[2 * in_stride];
    __m256 row3 = src[3 * in_stride];
    __m256 row4 = src[4 * in_stride];
    __m256 row5 = src[5 * in_stride];
    __m256 row6 = src[6 * in_stride];
    __m256 row7 = src[7 * in_stride];

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

inline void permute_8x8_devectorize2d(const vector<float, 8, 8> *src, float *dst,
                                 size_t in_stride, size_t out_stride) noexcept {
    __m256 row0 = src[0](in_stride);
    __m256 row1 = src[1](in_stride);
    __m256 row2 = src[2](in_stride);
    __m256 row3 = src[3](in_stride);
    __m256 row4 = src[4](in_stride);
    __m256 row5 = src[5](in_stride);
    __m256 row6 = src[6](in_stride);
    __m256 row7 = src[7](in_stride);

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

inline void permute_8x8_vectorize1d(const float *src, vector<float, 8> *dst,
                               size_t in_stride = 1,
                               size_t out_stride = 1) noexcept {
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

    __m256 u0 = _mm256_shuffle_ps(t0, t2, 0x44);
    __m256 u1 = _mm256_shuffle_ps(t0, t2, 0xEE);
    __m256 u2 = _mm256_shuffle_ps(t1, t3, 0x44);
    __m256 u3 = _mm256_shuffle_ps(t1, t3, 0xEE);
    __m256 u4 = _mm256_shuffle_ps(t4, t6, 0x44);
    __m256 u5 = _mm256_shuffle_ps(t4, t6, 0xEE);
    __m256 u6 = _mm256_shuffle_ps(t5, t7, 0x44);
    __m256 u7 = _mm256_shuffle_ps(t5, t7, 0xEE);

    dst[0 * out_stride] = _mm256_permute2f128_ps(u0, u4, 0x20);
    dst[1 * out_stride] = _mm256_permute2f128_ps(u1, u5, 0x20);
    dst[2 * out_stride] = _mm256_permute2f128_ps(u2, u6, 0x20);
    dst[3 * out_stride] = _mm256_permute2f128_ps(u3, u7, 0x20);
    dst[4 * out_stride] = _mm256_permute2f128_ps(u0, u4, 0x31);
    dst[5 * out_stride] = _mm256_permute2f128_ps(u1, u5, 0x31);
    dst[6 * out_stride] = _mm256_permute2f128_ps(u2, u6, 0x31);
    dst[7 * out_stride] = _mm256_permute2f128_ps(u3, u7, 0x31);
}

inline void permute_8x8_vectorize2d(const float *src, vector<float, 8, 8> *dst,
                               size_t in_stride = 1,
                               size_t out_stride = 1) noexcept {
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

    __m256 u0 = _mm256_shuffle_ps(t0, t2, 0x44);
    __m256 u1 = _mm256_shuffle_ps(t0, t2, 0xEE);
    __m256 u2 = _mm256_shuffle_ps(t1, t3, 0x44);
    __m256 u3 = _mm256_shuffle_ps(t1, t3, 0xEE);
    __m256 u4 = _mm256_shuffle_ps(t4, t6, 0x44);
    __m256 u5 = _mm256_shuffle_ps(t4, t6, 0xEE);
    __m256 u6 = _mm256_shuffle_ps(t5, t7, 0x44);
    __m256 u7 = _mm256_shuffle_ps(t5, t7, 0xEE);

    dst[0](out_stride) = _mm256_permute2f128_ps(u0, u4, 0x20);
    dst[1](out_stride) = _mm256_permute2f128_ps(u1, u5, 0x20);
    dst[2](out_stride) = _mm256_permute2f128_ps(u2, u6, 0x20);
    dst[3](out_stride) = _mm256_permute2f128_ps(u3, u7, 0x20);
    dst[4](out_stride) = _mm256_permute2f128_ps(u0, u4, 0x31);
    dst[5](out_stride) = _mm256_permute2f128_ps(u1, u5, 0x31);
    dst[6](out_stride) = _mm256_permute2f128_ps(u2, u6, 0x31);
    dst[7](out_stride) = _mm256_permute2f128_ps(u3, u7, 0x31);
}

// vectorize
template <> class u_vectorize<true, float, vector<float, 8>> {
  public:
    template <Dimension TM, Dimension TN, Dimension TMStrides>
    constexpr void operator()(const float *input, const TM &M, const TN &N,
                              const TMStrides &m_strides,
                              vector<float, 8> *output) noexcept {
        constexpr auto speedup_m = 8_dim;
        const bool speedup = M == speedup_m && N % 8 == 0 && m_strides != 1;

        if (speedup) {
            auto src = input;
            auto dst = output;
            for (size_t j = 0; j < N / speedup_m; j++) {
                permute_8x8_vectorize1d(src, dst, m_strides, 1);
                src += 8;
                dst += 8;
            }
        } else {
            ukernels::u_vectorize<false, float, vector<float, 8>> impl;
            impl(input, M, N, m_strides, output);
        }
    }
};

// vectorize2d
template <FixedTensor TIn, FixedTensor TOut>
class u_vectorize2d<true, TIn, TOut, float, vector<float, 8, 8>> {
  public:
    template <FixedDimensions TAxes>
    constexpr void operator()(const TIn &input, const TAxes &axes,
                              TOut &output) noexcept {

        constexpr auto axes_temp = TAxes{};
        constexpr auto conti_dims_input =
            contiguous_dims(input.shape(), input.strides());
        constexpr auto conti_dims_output =
            contiguous_dims(output.shape(), output.strides());

        if constexpr (TAxes::rank() == 2 &&
                      axes_temp[0_dim] + 1 == axes_temp[1_dim] &&
                      conti_dims_input == TIn::rank() &&
                      conti_dims_output == TOut::rank()) {

            if constexpr (TAxes::rank() > 0 &&
                          (TAxes{}[-1]) == (TIn::rank() - 1)) {
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
                    auto in_ptr = &input(in_index);
                    auto out_ptr = &output(out_index);
                    for (size_t i = 0; i < lanes[0]; i++) {
                        out_ptr[0](i) = _mm256_loadu_ps(in_ptr);
                        in_ptr += lanes[1] * out_shape[out_rank - 1];
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
                const auto vectorized_domain =
                    domain.template slice<axes_temp[0_dim], 2>();
                const auto inner_domain =
                    domain.template slice<axes_temp[1_dim] + 1>();
                const auto inner_size = inner_domain.length();

                if (inner_size % TVec::shape()[1_dim] != 0) {
                    ukernels::u_vectorize2d<false, TIn, TOut, float, TVec> impl;
                    impl(input, axes, output);
                } else {
                    ntt::apply(outer_domain, [&](auto index) {
                        loop<axes_temp[0_dim]>([&](auto i) {
                            inner_index[i] = index[i];
                            outer_index[i] = index[i];
                        });
                        for (size_t i = 0; i < vectorized_domain[0_dim]; i++) {
                            outer_index[axes[0_dim]] = i;
                            auto outer_ptr_keep = &output(outer_index);
                            for (size_t j = 0; j < lanes[0]; j++) {
                                inner_index[axes[0_dim]] = i * lanes[0_dim] + j;
                                auto outer_ptr = outer_ptr_keep;

                                for (size_t k = 0; k < vectorized_domain[1_dim];
                                     k++) {
                                    inner_index[axes[1_dim]] = k * lanes[1];
                                    auto input_ptr = &input(inner_index);

                                    for (size_t l = 0;
                                         l < inner_size / lanes[1_dim]; l++) {
                                        auto st_base = l * lanes[0_dim];
                                        auto ld_base = l * lanes[1_dim];

                                        auto src = input_ptr + ld_base;
                                        auto dst = outer_ptr + st_base;
                                        permute_8x8_vectorize2d(src, dst, inner_size,
                                                           j);
                                    }

                                    outer_ptr += inner_size;
                                }
                            }
                        }
                    });
                }
            }
        } else {
            using TElem = typename TIn::element_type;
            using TVec = typename std::decay_t<TOut>::element_type;
            ukernels::u_vectorize2d<false, TIn, TOut, TElem, TVec> impl;
            impl(input, axes, output);
        }
    }
};

template <Tensor TIn, Tensor TOut, size_t AxesRank>
    requires(
        (std::same_as<typename TIn::element_type, ntt::vector<float, 8, 8>> ||
         std::same_as<typename TIn::element_type, ntt::vector<float, 8>>) &&
        std::same_as<typename std::decay_t<TOut>::element_type, float> &&
        (AxesRank == 1 || AxesRank == 2))
class u_devectorize_impl<TIn, TOut, AxesRank, true> {
  public:
    using TVec = typename TIn::element_type;
    using TElem = typename std::decay_t<TOut>::element_type;

    template <FixedDimensions TAxes>
    constexpr void operator()(const TIn &input, TOut &output,
                              const TAxes &axes) {
        constexpr auto const_axes = TAxes{};
        if constexpr (AxesRank == 1) {
            if constexpr (const_axes[0] == (TIn::rank() - 1)) {
                auto size = output.size() * sizeof(TElem);
                auto in_ptr = input.buffer().data();
                auto out_ptr = output.buffer().data();
                std::memcpy(out_ptr, in_ptr, size);

            } else {
                constexpr auto in_rank = TIn::rank();
                constexpr auto axis = const_axes[0];
                auto in_shape = input.shape();

                dynamic_shape_t<in_rank> inner_domain;
                dynamic_shape_t<in_rank> domain;
                ntt::loop<in_rank>([&](auto &i) { domain[i] = in_shape[i]; });

                auto outer_index = domain.template slice<0, axis + 1>();
                auto inner_index =
                    domain.template slice<axis + 1, in_rank - (axis + 1)>();
                auto inner_size = inner_index.length();

                if (inner_size % 8 != 0) {
                    ukernels::u_devectorize_impl<TIn, std::decay_t<TOut>,
                                            TAxes::rank(), false>
                        impl;
                    impl(input, output, axes);
                } else {
                    auto out_ptr = &output(inner_domain);
                    auto dst = out_ptr;

                    ntt::apply(outer_index, [&](const auto &index) {
                        ntt::loop<axis + 1>(
                            [&](auto &i) { inner_domain[i] = index[i]; });
                        auto src = &input(inner_domain);

                        dst = out_ptr + dim_value(linear_offset(
                                            inner_domain, input.strides())) *
                                            8;

                        for (size_t i = 0; i < inner_size / 8; i++) {
                            permute_8x8_devectorize1d(src, dst, 1, inner_size);
                            dst += 8;
                            src += 8;
                        }
                    });
                }
            }
        } else if constexpr (const_axes[0] + 1 == const_axes[1]) {
            using TVec = vector<float, 8, 8>;
            constexpr auto const_axes = TAxes{};
            constexpr auto in_rank = TIn::rank();
            auto in_shape = input.shape();

            dynamic_shape_t<in_rank> domain;
            ntt::loop<in_rank>([&](auto &i) { domain[i] = in_shape[i]; });

            dynamic_shape_t<in_rank> inner_domain;

            auto vectorized_index = domain.template slice<const_axes[0], 2>();
            auto inner_index =
                domain.template slice<const_axes[1] + 1,
                                      in_rank - (const_axes[1] + 1)>();
            auto inner_size = inner_index.length();

            dynamic_shape_t<const_axes[1]> tile_domain;
            ntt::loop<const_axes[1]>(
                [&](auto &i) { tile_domain[i] = in_shape[i]; });

            auto out_ptr = &output(inner_domain);
            auto dst = out_ptr;

            if ((const_axes[1] == TIn::rank() - 1) &&
                const_axes[1] == const_axes[0] + 1) {
                const auto domain_input = input.shape();
                const auto out_stride = input.shape()[-1_dim] * 8;
                apply(domain_input, [&](auto index) {
                    dynamic_shape_t<in_rank> index_p1;
                    ntt::loop<in_rank>(
                        [&](auto &i) { index_p1[i] = index[i]; });
                    index_p1[-1_dim] = 0;
                    auto offset_vec =
                        linear_offset(index_p1, input.strides()) * 64;
                    auto offset_elem = index[-1_dim] * 8;
                    dst = out_ptr + offset_vec + offset_elem;

                    __m256 row0 = input(index)(0);
                    __m256 row1 = input(index)(1);
                    __m256 row2 = input(index)(2);
                    __m256 row3 = input(index)(3);
                    __m256 row4 = input(index)(4);
                    __m256 row5 = input(index)(5);
                    __m256 row6 = input(index)(6);
                    __m256 row7 = input(index)(7);
                    _mm256_storeu_ps(&dst[0 * out_stride], row0);
                    _mm256_storeu_ps(&dst[1 * out_stride], row1);
                    _mm256_storeu_ps(&dst[2 * out_stride], row2);
                    _mm256_storeu_ps(&dst[3 * out_stride], row3);
                    _mm256_storeu_ps(&dst[4 * out_stride], row4);
                    _mm256_storeu_ps(&dst[5 * out_stride], row5);
                    _mm256_storeu_ps(&dst[6 * out_stride], row6);
                    _mm256_storeu_ps(&dst[7 * out_stride], row7);
                });
            } else {
                if (inner_size % TVec::shape()[1] != 0) {
                    ukernels::u_devectorize_impl<TIn, std::decay_t<TOut>,
                                            TAxes::rank(), false>
                        impl;
                    impl(input, output, axes);
                } else {
                    ntt::apply(tile_domain, [&](auto index) {
                        ntt::loop<const_axes[1]>(
                            [&](auto &i) { inner_domain[i] = index[i]; });
                        auto src = &input(inner_domain);
                        dst = out_ptr +
                              linear_offset(inner_domain, input.strides()) * 64;
                        for (size_t i = 0; i < 8; i++) {
                            auto st_offset_i =
                                i * vectorized_index[1] * 8 * inner_size;
                            for (size_t j = 0; j < vectorized_index[1]; j++) {
                                auto st_offset_j = j * inner_size * 8;
                                auto ld_offset_j = src + j * inner_size;
                                auto st_offset =
                                    dst + st_offset_i + st_offset_j;
                                for (size_t k = 0; k < inner_size / 8; k++) {
                                    auto st_ptr = st_offset + k * 8;
                                    auto ld_ptr = ld_offset_j + k * 8;
                                    permute_8x8_devectorize2d(ld_ptr, st_ptr, i,
                                                         inner_size);
                                }
                            }
                        }
                    });
                }
            }
        } else {
            ukernels::u_devectorize_impl<TIn, std::decay_t<TOut>, TAxes::rank(),
                                    false>
                impl;
            impl(input, output, axes);
        }
    }
};

// reduce
template <reduce_op Op, class T> struct u_reduce_policy<Op, T, true> {
    static constexpr size_t unroll = 8;
};

// matmul
template <>
struct u_matmul_policy<matmul_vectorize_kind::no_vectorize, float, float, float, true> {
    static constexpr size_t m0_tile = 1;
    static constexpr size_t n0_tile = 1;
    static constexpr size_t m0_subtile = 0;
};

// Vectorize M
template <>
struct u_matmul_policy<matmul_vectorize_kind::vectorize_m, vector<float, 8>, float,
                       vector<float, 8>, true> {
    static constexpr size_t m0_tile = 2;
    static constexpr size_t n0_tile = 4;
    static constexpr size_t m0_subtile = 0;
};

// Vectorize K
template <>
struct u_matmul_policy<matmul_vectorize_kind::vectorize_k, vector<float, 8>,
                       vector<float, 8>, float, true> {
    static constexpr size_t m0_tile = 2;
    static constexpr size_t n0_tile = 2;
    static constexpr size_t m0_subtile = 0;
};

// Vectorize N
template <>
struct u_matmul_policy<matmul_vectorize_kind::vectorize_n, float, vector<float, 8>,
                       vector<float, 8>, true> {
    static constexpr size_t m0_tile = 4;
    static constexpr size_t n0_tile = 2;
    static constexpr size_t m0_subtile = 0;
};

template <>
struct u_matmul_m1_policy<matmul_vectorize_kind::vectorize_n, float, vector<float, 8>,
                          vector<float, 8>, true> {
    static constexpr size_t n0_tile = 7;
};

template <bool AccumulateC>
struct u_matmul<ukernels::matmul_vectorize_kind::vectorize_n, AccumulateC, false, false,
                1, 7, float, vector<float, 8>, vector<float, 8>, true> {
    template <class TA, class TB, class TC>
    constexpr void operator()(const TA &a, const TB &b, TC &c0,
                              size_t K) noexcept {
        NTT_ASSUME(K > 0);

        register __m256 c0_0 asm("ymm0") = {};
        register __m256 c0_1 asm("ymm1") = {};
        register __m256 c0_2 asm("ymm2") = {};
        register __m256 c0_3 asm("ymm3") = {};
        register __m256 c0_4 asm("ymm4") = {};
        register __m256 c0_5 asm("ymm5") = {};
        register __m256 c0_6 asm("ymm6") = {};

        if constexpr (AccumulateC) {
            c0_0 = c0(0, 0);
            c0_1 = c0(0, 1);
            c0_2 = c0(0, 2);
            c0_3 = c0(0, 3);
            c0_4 = c0(0, 4);
            c0_5 = c0(0, 5);
            c0_6 = c0(0, 6);
        }

        register __m256 a0_0 asm("ymm7");

        register __m256 b0_0 asm("ymm8");
        register __m256 b0_1 asm("ymm9");
        register __m256 b0_2 asm("ymm10");
        register __m256 b0_3 asm("ymm11");
        register __m256 b0_4 asm("ymm12");
        register __m256 b0_5 asm("ymm13");
        register __m256 b0_6 asm("ymm14");

        for (size_t k = 0; k < K; k++) {
            a0_0 = _mm256_broadcast_ss(&a(0, k));

            b0_0 = b(k, 0);
            b0_1 = b(k, 1);
            b0_2 = b(k, 2);
            b0_3 = b(k, 3);
            b0_4 = b(k, 4);
            b0_5 = b(k, 5);
            b0_6 = b(k, 6);

            c0_0 = _mm256_fmadd_ps(a0_0, b0_0, c0_0);
            c0_1 = _mm256_fmadd_ps(a0_0, b0_1, c0_1);
            c0_2 = _mm256_fmadd_ps(a0_0, b0_2, c0_2);
            c0_3 = _mm256_fmadd_ps(a0_0, b0_3, c0_3);
            c0_4 = _mm256_fmadd_ps(a0_0, b0_4, c0_4);
            c0_5 = _mm256_fmadd_ps(a0_0, b0_5, c0_5);
            c0_6 = _mm256_fmadd_ps(a0_0, b0_6, c0_6);
        }

        _mm256_store_ps((float *)&c0(0, 0), c0_0);
        _mm256_store_ps((float *)&c0(0, 1), c0_1);
        _mm256_store_ps((float *)&c0(0, 2), c0_2);
        _mm256_store_ps((float *)&c0(0, 3), c0_3);
        _mm256_store_ps((float *)&c0(0, 4), c0_4);
        _mm256_store_ps((float *)&c0(0, 5), c0_5);
        _mm256_store_ps((float *)&c0(0, 6), c0_6);
    }
};

// Vectorize MN
template <>
struct u_matmul_policy<matmul_vectorize_kind::vectorize_mn, vector<float, 8>,
                       vector<float, 8>, vector<float, 8, 8>, true> {
    static constexpr size_t m0_tile = 1;
    static constexpr size_t n0_tile = 2;
    static constexpr size_t m0_subtile = 4;
};

// Vectorize MK
template <>
struct u_matmul_policy<matmul_vectorize_kind::vectorize_mk, vector<float, 8, 8>,
                       vector<float, 8>, vector<float, 8>, true> {
    static constexpr size_t m0_tile = 1;
    static constexpr size_t n0_tile = 1;
    static constexpr size_t m0_subtile = 0;
};

// Vectorize KN
template <>
struct u_matmul_policy<matmul_vectorize_kind::vectorize_kn, vector<float, 8>,
                       vector<float, 8, 8>, vector<float, 8>, true> {
    static constexpr size_t m0_tile = 4;
    static constexpr size_t n0_tile = 2;
    static constexpr size_t m0_subtile = 0;
};

template <>
struct u_matmul_m1_policy<matmul_vectorize_kind::vectorize_kn, vector<float, 8>,
                          vector<float, 8, 8>, vector<float, 8>, true> {
    static constexpr size_t n0_tile = 4;
};

template <bool AccumulateC>
struct u_matmul<ukernels::matmul_vectorize_kind::vectorize_kn, AccumulateC, false, false,
                1, 4, vector<float, 8>, vector<float, 8, 8>, vector<float, 8>,
                true> {
    template <class TA, class TB, class TC>
    constexpr void operator()(const TA &a, const TB &b, TC &c0,
                              size_t K) noexcept {
        NTT_ASSUME(K > 0);

        register __m256 c0_0 asm("ymm0") = {};
        register __m256 c0_1 asm("ymm1") = {};
        register __m256 c0_2 asm("ymm2") = {};
        register __m256 c0_3 asm("ymm3") = {};

        if constexpr (AccumulateC) {
            c0_0 = c0(0, 0);
            c0_1 = c0(0, 1);
            c0_2 = c0(0, 2);
            c0_3 = c0(0, 3);
        }

        register __m256 a0_0 asm("ymm7");

        for (size_t k = 0; k < K; k++) {
            for (size_t sk = 0; sk < 8; sk++) {
                a0_0 = _mm256_broadcast_ss((const float *)&a(0, k) + sk);

                c0_0 = _mm256_fmadd_ps(a0_0, b(k, 0)(sk), c0_0);
                c0_1 = _mm256_fmadd_ps(a0_0, b(k, 1)(sk), c0_1);
                c0_2 = _mm256_fmadd_ps(a0_0, b(k, 2)(sk), c0_2);
                c0_3 = _mm256_fmadd_ps(a0_0, b(k, 3)(sk), c0_3);
            }
        }

        _mm256_store_ps((float *)&c0(0, 0), c0_0);
        _mm256_store_ps((float *)&c0(0, 1), c0_1);
        _mm256_store_ps((float *)&c0(0, 2), c0_2);
        _mm256_store_ps((float *)&c0(0, 3), c0_3);
    }
};

// Vectorize MKN
template <>
struct u_matmul_policy<matmul_vectorize_kind::vectorize_mkn, vector<float, 8, 8>,
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
