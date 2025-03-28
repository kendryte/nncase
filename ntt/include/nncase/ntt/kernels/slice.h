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
#include "../utility.h"

namespace nncase::ntt {
namespace slice_detail {
template <IsFixedDims TStart, IsFixedDims TStop, IsFixedDims TStride,
          IsFixedDims TAxes, IsFixedDims TShape, size_t... Ints>
inline constexpr auto compute_inner_domain(std::index_sequence<Ints...>) {
    return fixed_shape<(
        ((std::min(TShape::at(TAxes::at(Ints)), TStop::at(Ints)) - 1 -
          TStart::at(Ints)) /
         TStride::at(Ints)) +
        1)...>{};
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
template <typename TStart, typename TStop, typename TAxes, typename TStride,
          typename TIn, typename TOut>
void slice(const TIn &input, TOut &&output) {
    auto out_shape = output.shape();
    constexpr auto rank = out_shape.rank();
    auto count = out_shape[rank - 1];
    ranked_shape<rank> domain;
    loop<rank>([&](auto i) { domain[i] = out_shape[i]; });
    domain[rank - 1] = 1;

    // update starts/steps
    size_t in_starts[rank] = {0};
    size_t in_steps[rank] = {0};
    for (size_t i = 0; i < rank; i++) {
        in_steps[i] = 1;
    }
    loop<TAxes::rank()>([&](auto i) {
        in_starts[TAxes::at(i)] = TStart::at(i);
        in_steps[TAxes::at(i)] = TStride::at(i);
    });

    auto in_strides = input.strides();
    auto out_strides = output.strides();
    using element_type = typename TIn::element_type;
    apply(domain, [&](auto index) {
        auto pout =
            output.buffer().data() + linear_offset(index, output.strides());
        loop<rank>(
            [&](auto i) { index[i] = in_starts[i] + index[i] * in_steps[i]; });
        auto pin =
            input.buffer().data() + linear_offset(index, input.strides());
        u_unary<ntt::ops::copy<element_type>, element_type>(
            pin, in_strides[rank - 1] * in_steps[rank - 1], pout,
            out_strides[rank - 1], count);
    });
}
} // namespace nncase::ntt