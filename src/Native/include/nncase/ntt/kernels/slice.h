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
#include <iostream>
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

template <typename Tshape>
static void dump_shape(const std::string &info, Tshape shape) {
    std::cout << info;
    for (size_t i = 0; i < shape.rank(); i++)
        std::cout << shape[i] << " ";
    std::cout << std::endl;
}

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
#if 0
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
#else
    auto out_shape = output.shape();
    constexpr auto rank = out_shape.rank();
    auto count = out_shape[rank - 1];
    constexpr auto domain = shape_infer::reduced_shape_by_axes(
        typename std::decay_t<TOut>::shape_type{}, fixed_shape<rank - 1>{});

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

    auto out_step = output.strides()[rank - 1];
    apply(domain, [&](auto index) {
        auto pout =
            output.buffer().data() + linear_offset(index, output.strides());
        loop<rank>(
            [&](auto i) { index[i] = in_starts[i] + index[i] * in_steps[i]; });
        auto pin =
            input.buffer().data() + linear_offset(index, input.strides());
        u_slice(pin, in_steps[rank - 1], pout, out_step, count);
    });
#endif
}
} // namespace nncase::ntt