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
#include "../apply.h"
#include "../loop.h"
#include "../shape_infer/unpack.h"
#include "../ukernels.h"
#include <type_traits>

namespace nncase::ntt {
namespace detail {
template <Tensor TIn, Tensor TOut, size_t AxesRank> class unpack_impl {
  public:
    using TVec = typename TIn::element_type;

    template <FixedDimensions TAxes>
    constexpr void operator()(const TIn &input, TOut &output,
                              const TAxes &axes) {
        constexpr auto in_rank = TIn::rank();
        constexpr auto out_rank = TOut::rank();
        constexpr auto elem_rank = TVec::rank();
        constexpr auto elem_shape = TVec::shape();

        const auto conti_dims_input =
            contiguous_dims(input.shape(), input.strides());
        const auto conti_dims_output =
            contiguous_dims(output.shape(), output.strides());

        if (conti_dims_input == in_rank && conti_dims_output == out_rank &&
            AxesRank <= 2) {
            ntt::u_unpack(input, output, axes);
        } else {
            const auto domain = input.shape().concat(elem_shape);
            apply(domain, [&](auto index) {
                const auto in_index = index.template slice<0, in_rank>();
                const auto elem_index =
                    index.template slice<in_rank, elem_rank>();
                const auto out_index_template =
                    index.template slice<0, in_rank>();
                const auto out_index = axes.aggregate(
                    out_index_template,
                    [&](const auto cnt_out_index, auto axis, auto i) {
                        return cnt_out_index.template replace_at<axis>(
                            cnt_out_index[axis] * elem_shape[i] +
                            index[in_rank + i]);
                    });
                output(out_index) = input(in_index)(elem_index);
            });
        }
    }
};
#if 0
// Pack 1D
template <Tensor TIn, Tensor TOut> class unpack_impl<TIn, TOut, 1> {
  public:
    using TVec = typename TIn::element_type;

    template <FixedDimensions TAxes>
    constexpr void operator()(const TIn &input, TOut &output,
                              const TAxes &axes) {
        const auto PackAxis = axes[0_dim];
        constexpr auto rank = TIn::rank();
        constexpr auto elem_rank = TVec::shape_type::rank();
        constexpr auto elem_shape = TVec::shape();

        auto input_shape = input.shape();
        auto input_strides = input.strides();
        auto in_conti_dims = contiguous_dims(input_shape, input_strides);
        if (in_conti_dims == rank) {
            auto pin = input.buffer().data();
            auto pout = output.buffer().data();
            auto count = input.shape().length();
            if constexpr (PackAxis == (rank - 1))
                // FIXME: Unaligned copy
                ntt::u_unary(ntt::ops::copy<TVec>{}, pin, 1,
                             reinterpret_cast<TVec *>(pout), 1, count);
            else
                ntt::u_unpack_1d_ranked<elem_shape[0_dim], TVec,
                                        typename TOut::element_type>(
                    pin, 1,
                    input.shape().template slice<PackAxis + 1>().length(), pout,
                    count);
        } else {
            const auto domain = input.shape().concat(elem_shape);
            apply(domain, [&](auto index) {
                const auto in_index = index.template slice<0, rank>();
                const auto elem_index = index.template slice<rank, elem_rank>();
                const auto out_index_template = index.template slice<0, rank>();
                const auto out_index =
                    out_index_template.template replace_at<PackAxis>(
                        out_index_template[PackAxis] * elem_shape[0_dim] +
                        index[rank]);
                output(out_index) = input(in_index)(elem_index);
            });
        }
    }
};

// Pack 2D
template <Tensor TIn, Tensor TOut> class unpack_impl<TIn, TOut, 2> {
  public:
    using TVec = typename TIn::element_type;

    template <FixedDimensions TAxes>
    constexpr void operator()(const TIn &input, TOut &output,
                              const TAxes &axes) {
        constexpr auto PackAxis0 = axes[0_dim];
        constexpr auto PackAxis1 = axes[1_dim];
        constexpr auto rank = TIn::rank();
        constexpr auto elem_rank = TVec::shape_type::rank();
        constexpr auto elem_shape = TVec::shape();

        auto input_shape = input.shape();
        auto input_strides = input.strides();
        auto in_conti_dims = contiguous_dims(input_shape, input_strides);
        if ((in_conti_dims == rank) && (PackAxis1 == PackAxis0 + 1) &&
            (PackAxis1 != (rank - 1))) {
            auto pout = output.buffer().data();
            auto count = input.shape().length();
            ntt::u_unpack_2d_ranked<elem_shape[0_dim], elem_shape[1_dim], TIn,
                                    typename TOut::element_type, PackAxis0,
                                    PackAxis1>(
                input, 1, input_shape.template slice<PackAxis0 + 1>().length(),
                input_shape.template slice<PackAxis1 + 1>().length(), pout,
                count);
        } else {
            const auto domain = input.shape().concat(elem_shape);
            apply(domain, [&](auto index) {
                const auto in_index = index.template slice<0, rank>();
                const auto elem_index = index.template slice<rank, elem_rank>();
                const auto out_index_template = index.template slice<0, rank>();
                const auto out_index = axes.aggregate(
                    out_index_template,
                    [&](const auto cnt_out_index, auto axis, auto i) {
                        return cnt_out_index.template replace_at<axis>(
                            cnt_out_index[axis] * elem_shape[i] +
                            index[rank + i]);
                    });
                output(out_index) = input(in_index)(elem_index);
            });
        }
    }
};
#endif
} // namespace detail

template <Tensor TIn, class TOut, FixedDimensions TAxes>
void unpack(const TIn &input, TOut &&output, const TAxes &axes) noexcept {
    detail::unpack_impl<TIn, std::decay_t<TOut>, TAxes::rank()> impl;
    impl(input, output, axes);
}
} // namespace nncase::ntt
