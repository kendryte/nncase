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
#include "../tensor_traits.h"
#include "../ukernels/u_pack.h"

namespace nncase::ntt {
namespace detail {
template <Tensor TIn, Tensor TOut, size_t AxesRank> class pack_impl {
  public:
    using TElem = typename TIn::element_type;
    using TVec = typename std::decay_t<TOut>::element_type;

    template <FixedDimensions TAxes>
    constexpr void operator()(const TIn &input, TOut &output,
                              const TAxes &axes) {
        constexpr auto in_rank = TIn::rank();
        constexpr auto out_rank = TOut::rank();
        constexpr auto lanes = TVec::shape();

        const auto conti_dims_input =
            contiguous_dims(input.shape(), input.strides());
        const auto conti_dims_output =
            contiguous_dims(output.shape(), output.strides());

        if (TAxes::rank() == 2 && axes[0_dim] + 1 == axes[1_dim] &&
            conti_dims_input == in_rank && conti_dims_output == out_rank) {
            ntt::u_pack2d(input, axes, output);
        } else {
            const auto domain = output.shape().concat(lanes);
            apply(domain, [&](auto index) {
                const auto out_index = index.template slice<0, out_rank>();
                const auto in_index_template =
                    index.template slice<0, in_rank>();
                const auto elem_index = index.template slice<out_rank>();

                bool skip = false;
                const auto in_index = axes.aggregate(
                    in_index_template,
                    [&](const auto &cnt_in_index, auto axis, auto i) {
                        const auto in_dim =
                            cnt_in_index[axis] * lanes[i] + index[out_rank + i];
                        if (in_dim >= input.shape()[axis]) {
                            skip = true;
                        }
                        return cnt_in_index.template replace_at<axis>(in_dim);
                    });
                output(out_index)(elem_index) =
                    skip ? (TElem)0 : input(in_index);
            });
        }
    }
};

// 1D packing
template <Tensor TIn, Tensor TOut> class pack_impl<TIn, TOut, 1> {
  public:
    using TVec = typename std::decay_t<TOut>::element_type;

    static inline constexpr auto VecLen = fixed_dim_v<TVec::shape().length()>;

    template <FixedDimensions TAxes>
    constexpr void operator()(const TIn &input, TOut &output,
                              const TAxes &axes) {
        const auto PackAxis = axes[0_dim];
        auto in_p = input.elements().data();
        auto out_p = output.elements().data();
        constexpr auto rank = TIn::rank();
        const auto in_shape = input.shape();
        const auto in_conti_dims = contiguous_dims(in_shape, input.strides());
        const auto out_conti_dims =
            contiguous_dims(output.shape(), output.strides());
        if ((in_conti_dims == rank) && (out_conti_dims == rank) &&
            (PackAxis == rank - 1) && (in_shape.length() % VecLen == 0)) {
            for (dim_t i = 0; i < output.shape().length(); i++) {
                out_p[i] = TVec::unaligned_load_from(in_p);
                in_p += VecLen;
            }
        } else {
            apply<0, PackAxis>(input, output, in_p, out_p);
        }
    }

  private:
    template <size_t Axis, size_t PackAxis, class TInP, class TOutP>
    constexpr void apply(const TIn &input, TOut &output, TInP in_p,
                         TOutP out_p) {
        const auto axis_v = fixed_dim_v<Axis>;
        if constexpr (Axis < PackAxis) {
            for (size_t i = 0; i < output.shape()[Axis]; i++) {
                apply<Axis + 1, PackAxis>(input, output, in_p, out_p);
                in_p += input.strides()[axis_v];
                out_p += output.strides()[axis_v];
            }
        } else {
            constexpr auto rest_rank = TIn::rank() - axis_v - 1;
            const auto conti_dims = ntt::min(
                rest_rank, contiguous_dims(input.shape(), input.strides()),
                contiguous_dims(input.shape(), input.strides()));
            const auto m_strides = input.strides()[axis_v];

            for (size_t i = 0; i < input.shape()[axis_v] / VecLen; i++) {
                apply_transpose<Axis + 1>(input, conti_dims, VecLen, m_strides,
                                          output, in_p, out_p);

                in_p += input.strides()[axis_v] * VecLen;
                out_p += output.strides()[axis_v];
            }

            // Tail
            const auto tail_m = input.shape()[axis_v] % VecLen;
            if (tail_m) {
                apply_transpose<Axis + 1>(input, conti_dims, tail_m, m_strides,
                                          output, in_p, out_p);
            }
        }
    }

    template <size_t Axis, class TInP, Dimension TContiguousDims, Dimension TM,
              Dimension TMStrides, class TOutP>
    constexpr void apply_transpose(const TIn &input,
                                   const TContiguousDims &conti_dims,
                                   const TM &M, const TMStrides &m_strides,
                                   TOut &output, TInP in_p, TOutP out_p) {
        const auto axis_v = fixed_dim_v<Axis>;
        if (Axis + conti_dims == TOut::rank()) {
            constexpr auto rest_rank = TOut::rank() - axis_v;
            const auto rest_dims =
                output.shape().template slice<TOut::rank() - rest_rank>();
            const auto N = rest_dims.length();
            ntt::u_pack(in_p, M, N, m_strides, out_p);
        } else if constexpr (Axis + 1 < TOut::rank()) {
            for (size_t i = 0; i < output.shape()[axis_v]; i++) {
                apply_transpose<Axis + 1>(input, conti_dims, M, m_strides,
                                          output, in_p, out_p);
                in_p += input.strides()[axis_v];
                out_p += output.strides()[axis_v];
            }
        }
    }
};
} // namespace detail

template <Tensor TIn, class TOut, FixedDimensions TAxes>
void pack(const TIn &input, TOut &&output, const TAxes &axes) noexcept {
    using TVec = typename std::decay_t<TOut>::element_type;
    static_assert(TVec::rank() == TAxes::rank(),
                  "Output vector rank must match axes rank");
    detail::pack_impl<TIn, std::decay_t<TOut>, TAxes::rank()> impl;
    impl(input, output, axes);
}
} // namespace nncase::ntt
