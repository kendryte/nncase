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
#include "reduce.h"

namespace nncase::ntt {

namespace vectorized_layer_norm_detail {

template <Tensor TIn, Tensor TScale, Tensor TBias, typename TOut,
          FixedDimensions VectorizedAxes, Dimensions PadedNums,
          FixedDimension TAxis>
void within_axis_vectorize_impl(const TIn &input, const TScale &scale,
                                const TBias &bias, TOut &output,
                                const float &epsilon, const VectorizedAxes &,
                                const PadedNums &, const TAxis &,
                                const bool use_mean = true) {

    using TElem = typename TIn::element_type;
    using TAccElem = decltype(ntt::cast_elem<float>(std::declval<TElem>()));

    auto input_shape = input.shape();
    auto input_strides = input.strides();

    constexpr auto axis_value = positive_index(TAxis::value, TIn::rank());
    const auto domain =
        input_shape.template slice<(size_t)0, (size_t)axis_value>();
    const auto strides =
        input_strides.template slice<(size_t)0, (size_t)axis_value>();

    const auto inner_size =
        input_shape.template slice<(size_t)axis_value>().length();

    constexpr VectorizedAxes vectorized_axes_temp;
    constexpr bool UseVectorReduce = vectorized_axes_temp.rank() == 1 &&
                                     vectorized_axes_temp[0] >= axis_value;

    using TElemScalar = element_or_scalar_t<TElem>;
    auto finner_size = (float)inner_size;
    if constexpr (UseVectorReduce) {
        finner_size *= TElem::size();
    }
    const auto norm_factor = 1.f / finner_size;

    ntt::apply(domain, [&](auto index) {
        const auto input_p =
            input.elements().data() + linear_offset(index, strides);
        const auto scale_p = scale.elements().data();
        const auto bias_p = bias.elements().data();
        auto output_p =
            output.elements().data() + linear_offset(index, strides);

        if constexpr (UseVectorReduce) {
            auto mean = (TElemScalar)0;
            auto extended_sum = (TAccElem)0;
            if (use_mean) {
                auto extended_mean = (TAccElem)0;
                for (size_t i = 0; i < inner_size; i++)
                    extended_mean += ntt::cast_elem<float>(input_p[i]);
                extended_mean *= norm_factor;
                mean = (TElemScalar)reduce_sum(extended_mean);

                for (auto i = 0; i < inner_size; i++) {
                    const auto val = ntt::square(input_p[i] - mean);
                    extended_sum += ntt::cast_elem<float>(val);
                }
            } else {
                for (auto i = 0; i < inner_size; i++) {
                    const auto val = ntt::square(input_p[i]);
                    extended_sum += ntt::cast_elem<float>(val);
                }
            }

            const auto extended_sum_s = reduce_sum(extended_sum) * norm_factor;
            auto extended_add = extended_sum_s + epsilon;
            auto rsqrt = ntt::cast_elem<TElemScalar>(ntt::rsqrt(extended_add));

            if (use_mean) {
                for (auto i = 0; i < inner_size; i++) {
                    auto val = (input_p[i] - mean) * rsqrt;
                    output_p[i] = val * scale_p[i] + bias_p[i];
                }
            } else {
                for (auto i = 0; i < inner_size; i++) {
                    auto val = input_p[i] * rsqrt;
                    output_p[i] = val * scale_p[i] + bias_p[i];
                }
            }
        } else {
            auto mean = (TElem)0;
            auto extended_sum = (TAccElem)0;
            if (use_mean) {
                auto extended_mean = (TAccElem)0;
                for (size_t i = 0; i < inner_size; i++)
                    extended_mean += ntt::cast_elem<float>(input_p[i]);
                extended_mean *= norm_factor;
                mean = ntt::cast_elem<TElemScalar>(extended_mean);

                for (auto i = 0; i < inner_size; i++) {
                    const auto val = ntt::square(input_p[i] - mean);
                    extended_sum += ntt::cast_elem<float>(val);
                }
            } else {
                for (auto i = 0; i < inner_size; i++) {
                    const auto val = ntt::square(input_p[i]);
                    extended_sum += ntt::cast_elem<float>(val);
                }
            }

            extended_sum *= norm_factor;
            auto extended_add = extended_sum + epsilon;
            auto rsqrt = ntt::cast_elem<TElemScalar>(ntt::rsqrt(extended_add));

            if (use_mean) {
                for (auto i = 0; i < inner_size; i++) {
                    auto val = (input_p[i] - mean) * rsqrt;
                    output_p[i] = val * scale_p[i] + bias_p[i];
                }
            } else {
                for (auto i = 0; i < inner_size; i++) {
                    auto val = input_p[i] * rsqrt;
                    output_p[i] = val * scale_p[i] + bias_p[i];
                }
            }
        }
    });
}

} // namespace vectorized_layer_norm_detail

template <Tensor TIn, Tensor TScale, Tensor TBias, typename TOut,
          FixedDimension TAxis, FixedDimensions VectorizedAxes = shape_t<>,
          Dimensions PadedNums = shape_t<>>
void vectorized_layer_norm(const TIn &input, const TScale &scale,
                           const TBias &bias, TOut &&output,
                           const float &epsilon, const TAxis &axis = -1_dim,
                           const VectorizedAxes &vectorizedAxes = {},
                           const PadedNums &padedNums = {},
                           const bool use_mean = true) {
    static_assert(VectorizedAxes::rank() < 2,
                  "currently not support 2d packing.");

    vectorized_layer_norm_detail::within_axis_vectorize_impl<
        TIn, TScale, TBias, TOut, VectorizedAxes, PadedNums, TAxis>(
        input, scale, bias, output, epsilon, vectorizedAxes, padedNums, axis,
        use_mean);
}
} // namespace nncase::ntt
