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
#include "reduce.h"

namespace nncase::ntt {

namespace packed_layer_norm_detail {

template <Tensor TIn, Tensor TScale, Tensor TBias, typename TOut, Scalar TEp,
          FixedDimensions PackedAxes, FixedDimensions PadedNums,
          FixedDimensions TAxis = shape_t<>>
void within_axis_pack_impl(const TIn &input, const TScale &scale,
                           const TBias &bias, TOut &&output, const TEp &epsilon,
                           const PackedAxes &, const PadedNums &,
                           const TAxis &) {

    using TElem = typename TIn::element_type;
    auto input_shape = input.shape();
    auto input_strides = input.strides();

    TAxis axis;
    constexpr auto axis_value = axis[0_dim];
    const auto domain = input_shape.template slice<0, axis_value>();
    const auto strides = input_strides.template slice<0, axis_value>();

    const auto inner_size =
        input_shape.template slice<axis_value, input.rank() - axis_value>()
            .length();

    constexpr PackedAxes packed_axes_temp;
    constexpr bool UseVectorReduce =
        packed_axes_temp.rank() == 1 && packed_axes_temp[0] >= axis_value;

    TElem finner_size = (TElem)inner_size;
    if constexpr (UseVectorReduce) {
        finner_size = finner_size * (TElem)TElem::size();
    }

    constexpr bool use_mean = true; // This can be a parameter if needed
    apply(domain, [&](auto index) {
        const auto input_p =
            input.elements().data() + linear_offset(index, strides);
        const auto scale_p = scale.elements().data();
        const auto bias_p = bias.elements().data();
        auto output_p =
            output.elements().data() + linear_offset(index, strides);

        // compute mean
        TElem mean1 = (TElem)0;
        if constexpr (use_mean) {
            for (size_t i = 0; i < inner_size; i++)
                mean1 = mean1 + (input_p[i] / finner_size);
            if constexpr (UseVectorReduce) {
                mean1 = (TElem)reduce_sum(mean1);
            }
        }

        for (auto i = 0; i < inner_size; i++)
            output_p[i] = input_p[i] - mean1;

        for (auto i = 0; i < inner_size; i++)
            output_p[i] = output_p[i] * output_p[i];

        TElem mean2 = (TElem)0;
        for (auto i = 0; i < inner_size; i++)
            mean2 = mean2 + (output_p[i] / finner_size);
        if constexpr (UseVectorReduce) {
            mean2 = (TElem)reduce_sum(mean2);
        }

        TElem add = mean2 + epsilon;
        TElem sqrt = ntt::sqrt(add);

        for (auto i = 0; i < inner_size; i++)
            output_p[i] = (input_p[i] - mean1) / sqrt;

        for (auto i = 0; i < inner_size; i++)
            output_p[i] = (output_p[i] * (TElem)scale_p[i]) + (TElem)bias_p[i];
    });
}

} // namespace packed_layer_norm_detail

template <Tensor TIn, Tensor TScale, Tensor TBias, typename TOut, Scalar TEp,
          FixedDimensions PackedAxes = shape_t<>,
          FixedDimensions PadedNums = shape_t<>,
          FixedDimensions TAxis = shape_t<>>
void packed_layer_norm(const TIn &input, const TScale &scale, const TBias &bias,
                       TOut &&output, const TEp &epsilon,
                       const PackedAxes &packedAxes, const PadedNums &padedNums,
                       const TAxis &axes) {
    static_assert(PackedAxes::rank() < 2, "currently not support 2d packing.");
    if constexpr (PackedAxes::rank() <= 1) {
        static_assert(PadedNums::rank() == 0, "not support padding");
    }

    packed_layer_norm_detail::within_axis_pack_impl<
        TIn, TScale, TBias, TOut, TEp, PackedAxes, PadedNums, TAxis>(
        input, scale, bias, output, epsilon, packedAxes, padedNums, axes);
}
} // namespace nncase::ntt
