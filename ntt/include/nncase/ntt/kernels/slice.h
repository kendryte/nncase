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
#include "../primitive_ops.h"
#include "../ukernels.h"
#include "../utility.h"
#include <tuple>

namespace nncase::ntt {
namespace slice_detail {
template <Dimension TX, Dimension TDim, Dimension TLowerBound,
          Dimension TUpperBound>
constexpr auto translate_begin_end(const TX &x, const TDim &dim,
                                   const TLowerBound &lower_bound,
                                   const TUpperBound &upper_bound) {
    const auto new_x = positive_index(x, dim);
    return ntt::clamp(new_x, lower_bound, dim + upper_bound);
}

template <Shape TInShape, Dimensions TBegins, FixedDimensions TAxes,
          FixedDimensions TSteps>
constexpr auto translate_slice_params(const TInShape &in_shape,
                                      const TBegins &begins, const TAxes &axes,
                                      const TSteps &steps) {
    constexpr auto rank = TInShape::rank();
    const auto new_begins =
        axes.aggregate(make_zeros_shape<rank>(), [&](const auto cnt_new_begins,
                                                     auto axis, auto i) {
            const auto in_dim = in_shape[axis];
            return cnt_new_begins.template replace_at<axis>(ntt::select(
                steps[i] < dim_zero,
                // for negative step: begins[i] is clamped into the range
                // [0, dims[axes[i]]-1].
                translate_begin_end(begins[i], in_dim, 0_dim, -1_dim),
                // for positive step: begins[i] is clamped into the range
                // [0, dims[axes[i]]].
                translate_begin_end(begins[i], in_dim, 0_dim, 0_dim)));
        });
    const auto new_steps =
        axes.aggregate(make_ones_shape<rank>(), [&](const auto cnt_new_steps,
                                                    auto axis, auto i) {
            return cnt_new_steps.template replace_at<axis>(steps[i]);
        });
    return std::make_tuple(new_begins, new_steps);
}
} // namespace slice_detail

/**
 * @brief
 *
 * @tparam TStart start.
 * @tparam TStop stop.
 * @tparam TAxes axes.
 * @tparam TStride stride.
 * @param input input tensor
 * @param output output tensor
 */
template <Tensor TIn, typename TOut, typename TBegins, typename TEnds,
          FixedDimensions TAxes = decltype(make_index_shape<TBegins::rank()>()),
          FixedDimensions TSteps = decltype(make_ones_shape<TBegins::rank()>())>
void slice(const TIn &input, TOut &&output, const TBegins &begins,
           [[maybe_unused]] const TEnds &ends, const TAxes &axes = {},
           const TSteps &steps = {}) {
    static_assert(TBegins::rank() == TEnds::rank() &&
                      TBegins::rank() == TAxes::rank() &&
                      TBegins::rank() == TSteps::rank(),
                  "begins, ends, axes and steps must have the same "
                  "rank");
    auto [new_begins, new_steps] = slice_detail::translate_slice_params(
        input.shape(), begins, axes, steps);
    const auto out_shape = output.shape();
    constexpr auto rank = out_shape.rank();
    const auto count = out_shape[-1_dim];
    const auto domain = out_shape.template slice<0, rank - 1>().append(1_dim);

    auto in_strides = input.strides();
    auto out_strides = output.strides();
    using element_type = typename TIn::element_type;
    apply(domain, [&, new_begins = new_begins,
                   new_steps = new_steps](auto out_index) {
        auto pout =
            output.buffer().data() + linear_offset(out_index, output.strides());
        const auto in_index = generate_shape<rank>(
            [&, new_begins = new_begins, new_steps = new_steps](auto i) {
                return new_begins[i] + out_index[i] * new_steps[i];
            });
        auto pin =
            input.buffer().data() + linear_offset(in_index, input.strides());
        ntt::u_unary(ntt::ops::copy<element_type>{}, pin,
                     in_strides[-1_dim] * new_steps[-1_dim], pout,
                     out_strides[-1_dim], count);
    });
}
} // namespace nncase::ntt
