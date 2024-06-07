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
#include "../shape_infer/reduce_axis.h"
#include "../utility.h"
#include <tuple>

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
template <IsFixedDims TStart, IsFixedDims TStop, IsFixedDims TAxes,
          IsFixedDims TStride, IsFixedTensor TIn, IsFixedTensor TOut>
void slice(const TIn &input, TOut &&output) {
    constexpr auto domain = shape_infer::reduced_shape_by_axes(
        typename std::decay_t<TOut>::shape_type{}, TAxes{});
    constexpr auto inner_domain =
        slice_detail::compute_inner_domain<TStart, TStop, TStride, TAxes,
                                           typename TIn::shape_type>(
            std::make_index_sequence<TAxes::rank()>{});

    auto in_index = ranked_shape<domain.rank()>{};
    auto out_index = ranked_shape<domain.rank()>{};
    apply(domain, [&](auto index) {
        loop<domain.rank()>([&](auto i) {
            in_index[i] = index[i];
            out_index[i] = index[i];
        });

        apply(inner_domain, [&](auto inner_index) {
            loop<inner_domain.rank()>([&](auto i) {
                in_index[TAxes::at(i)] =
                    TStart::at(i) + inner_index[i] * TStride::at(i);
                out_index[TAxes::at(i)] = inner_index[i];
            });
            output(out_index) = input(in_index);
        });
    });
}
} // namespace nncase::ntt