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
#include <cstdint>

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

template <class TInShape, class TBegins, class TEnds, class TStrides,
          class TAxes>
auto slice_fill(const TInShape &in_shape, const TBegins &begins_value,
                const TEnds &ends_value, const TStrides &strides_value,
                const TAxes &axes_value) {
    constexpr auto ndim = in_shape.rank();
    using dims_t = std::array<int64_t, ndim>;
    dims_t begin_values{};
    dims_t end_values{};
    std::copy(in_shape.begin(), in_shape.end(), end_values.begin());
    dims_t strides_values{};
    strides_values.fill(1);
    for (size_t i = 0; i < ndim; ++i) {
        const auto it = std::find_if(axes_value.begin(), axes_value.end(),
                                     [i, ndim](const auto axis) {
                                         return positive_index(axis, ndim) == i;
                                     });
        if (it != axes_value.end()) {
            auto idx = (size_t)std::distance(axes_value.begin(), it);
            auto max = static_cast<int64_t>(in_shape[i]);
            auto min = (-1) * max - 1;

            // check starts
            begin_values[i] = begins_value(idx) < min   ? min
                              : begins_value(idx) > max ? max
                                                        : begins_value(idx);

            // check stops
            end_values[i] = ends_value(idx) < min   ? min
                            : ends_value(idx) > max ? max
                                                    : ends_value(idx);

            // check steps
            if (strides_value.rank()) {
                strides_values[i] = strides_value[idx];
            }

            // fixup begin_values
            if ((strides_values[i] > 0 && end_values[i] > begin_values[i]) ||
                (strides_values[i] < 0 && end_values[i] < begin_values[i])) {
                begin_values[i] =
                    begin_values[i] == min ? min + 1 : begin_values[i];
                begin_values[i] =
                    begin_values[i] == max ? max - 1 : begin_values[i];
            }
            if (begin_values[i] < 0)
                begin_values[i] += max;
            if (end_values[i] < 0)
                end_values[i] += max;
        }
    }
    return std::tuple(begin_values, end_values, strides_values);
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
template <typename TAxes, typename TStrides, typename TIn, typename TBegins,
          typename TEnds, typename TOut>
void slice(const TIn &input, const TBegins &begins, const TEnds &ends,
           TOut &&output) {
    auto &&[begin_values, end_values, strides_values] =
        slice_detail::slice_fill(input.shape(), begins, ends, TStrides{},
                                 TAxes{});
    apply(input.shape(), [&](auto in_index) {
        auto out_index = in_index;
        for (size_t i = 0; i < TIn::rank(); i++) {
            const auto stride = strides_values[i];
            if (stride > 0) {
                if ((int64_t)in_index[i] < begin_values[i] ||
                    in_index[i] >= static_cast<size_t>(end_values[i]))
                    return;
            } else {
                if ((int64_t)in_index[i] <= end_values[i] ||
                    (int64_t)in_index[i] > begin_values[i])
                    return;
            }

            auto out_div =
                std::div((int64_t)in_index[i] - begin_values[i], stride);
            if (out_div.rem)
                return;
            out_index[i] = (size_t)out_div.quot;
        }

        output(out_index) = input(in_index);
    });
}
} // namespace nncase::ntt
