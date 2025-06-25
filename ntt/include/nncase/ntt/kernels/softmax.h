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
#include "../shape_infer/reduce_axis.h"
#include "reduce.h"

namespace nncase::ntt {
namespace softmax_detail {
template <Tensor TIn, class TOut, FixedDimension TAxis,
          FixedDimensions PackedAxes>
void packed_softmax_impl(const TIn &input, TOut &&output, const TAxis &axis,
                         const PackedAxes &) {
    using TElem = typename TIn::element_type;
    auto input_shape = input.shape();

    constexpr PackedAxes packed_axes;
    constexpr auto need_reduce =
        PackedAxes::rank() != 0 && TAxis::value == packed_axes[0];
    auto domain = shape_infer::reduced_shape_by_axis<TAxis::value>(input_shape);
    apply(domain, [&](auto index) {
        // max
        TElem max_value = input(index);
        for (index[axis] = 0; index[axis] < input_shape.at(axis);
             index[axis]++) {
            max_value = ntt::max(max_value, input(index));
        }

        // reduce_max
        if constexpr (need_reduce) {
            max_value = (TElem)reduce_max(max_value);
        }

        // (x - reduce_max) * beta
        for (index[axis] = 0; index[axis] < input_shape.at(axis);
             index[axis]++) {
            output(index) = input(index) - max_value;
        }

        // exp((x - reduce_max) * beta) and sum
        TElem sum = (TElem)0;
        for (index[axis] = 0; index[axis] < input_shape.at(axis);
             index[axis]++) {
            output(index) = exp(output(index));
            sum += output(index);
        }

        // reduce sum
        if constexpr (need_reduce) {
            sum = (TElem)reduce_sum(sum);
        }

        // div
        for (index[axis] = 0; index[axis] < input_shape.at(axis);
             index[axis]++) {
            output(index) = output(index) / sum;
        }
    });
}

} // namespace softmax_detail

/**
 * @brief packed softmax
 *  implement notice:
 *    1. need support 2d pack.
 *    2. need support paded nums.
 *    3. need different implementation when the packed axis is equal or not
 * equal axis.
 * @tparam Axis softmax reduced axis
 * @param input input tensor.
 * @param output output output.
 * @param packedAxes  packed axes
 */
template <Tensor TIn, class TOut, FixedDimension TAxis,
          FixedDimensions PackedAxes = shape_t<>>
void packed_softmax(const TIn &input, TOut &&output, const TAxis &axis,
                    const PackedAxes &packedAxes = {}) noexcept {
    static_assert(PackedAxes::rank() < 2, "currently not support 2d pack");
    softmax_detail::packed_softmax_impl(input, output, axis, packedAxes);
}
} // namespace nncase::ntt
