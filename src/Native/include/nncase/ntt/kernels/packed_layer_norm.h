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
#include "../vector_ops.h"
#include "binary.h"
#include "unary.h"

namespace nncase::ntt {

namespace packed_layer_norm_detail {

template <size_t Axis, IsFixedTensor TIn, IsFixedTensor TScale,
          IsFixedTensor TBias, IsFixedTensor TOut, typename TEp>
void within_axis_pack_impl(const TIn &input, const TScale &scale,
                           const TBias &bias, TOut &&output, const TEp &epsilon,
                           [[maybe_unused]] const bool &use_mean) {
    using TElem = TIn::element_type;
    using TScalar = typename TIn::element_type::element_type;
    constexpr auto input_shape = typename TIn::shape_type{};
    constexpr auto input_strides = typename TIn::strides_type{};
    constexpr auto scale_shape = typename TScale::shape_type{};
    constexpr auto scale_strides = typename TScale::strides_type{};
    constexpr auto bias_shape = typename TBias::shape_type{};
    constexpr auto bias_strides = typename TBias::strides_type{};
    constexpr auto output_shape = typename std::decay_t<TOut>::shape_type{};
    constexpr auto output_strides = typename std::decay_t<TOut>::strides_type{};
    constexpr size_t in_contigous_dim =
        contiguous_dims(input_shape, input_strides);
    constexpr size_t scale_contiguous_dims =
        contiguous_dims(scale_shape, scale_strides);
    constexpr size_t bias_contiguous_dims =
        contiguous_dims(bias_shape, bias_strides);
    constexpr size_t output_contiguous_dims =
        contiguous_dims(output_shape, output_strides);
    static_assert(in_contigous_dim != 0 || scale_contiguous_dims != 0 ||
                      bias_contiguous_dims != 0 || output_contiguous_dims != 0,
                  "currently not support no contiguous!");
    static_assert(is_same_seq(input_shape, output_shape), "shape not match");
    static_assert(is_same_seq(input_strides, output_strides),
                  "strides not match");
    constexpr auto domain = slice_fixed_dims<Axis>(input_shape);
    constexpr auto strides = slice_fixed_dims<Axis>(input_strides);
    constexpr size_t inner_size =
        slice_fixed_dims<input_shape.rank() - Axis, Axis>(input_shape).length();
    constexpr auto sqrt_op = mathops::sqrt<TElem>();
    constexpr auto div_op = mathops::div<TElem>();
    constexpr auto sub_op = mathops::sub<TElem>();
    constexpr auto add_op = mathops::add<TElem>();
    constexpr auto mul_op = mathops::mul<TElem>();
    constexpr auto vsum_op = vector_ops::reduce_sum<TElem>();

    TElem finner_size = inner_size * TElem::shape_type::length();

    apply(domain, [&](auto index) {
        auto input_p = input.buffer().data() + linear_offset(index, strides);
        auto scale_p = scale.buffer().data();
        auto bias_p = bias.buffer().data();
        auto output_p = output.buffer().data() + linear_offset(index, strides);

        // compute mean
        TElem mean1 = 0;
        if (use_mean) {
            for (size_t i = 0; i < inner_size; i++)
                mean1 = add_op(mean1, div_op(input_p[i], finner_size));
            mean1 = vsum_op(mean1);
        }

        std::array<TElem, inner_size> sub;
        for (auto i = 0; i < inner_size; i++)
            sub[i] = sub_op(input_p[i], mean1);

        std::array<TElem, inner_size> pow;
        for (auto i = 0; i < inner_size; i++)
            pow[i] = mul_op(sub[i], sub[i]);

        TElem mean2 = 0;
        for (auto i = 0; i < inner_size; i++)
            mean2 = add_op(mean2, div_op(pow[i], finner_size));
        mean2 = vsum_op(mean2);

        TElem add = add_op(mean2, epsilon);
        TElem sqrt = sqrt_op(add);

        std::array<TElem, inner_size> norm;
        for (auto i = 0; i < inner_size; i++)
            norm[i] = div_op(sub[i], sqrt);

        for (auto i = 0; i < inner_size; i++)
            output_p[i] = add_op(mul_op(norm[i], scale_p[i]), bias_p[i]);
    });
}
} // namespace packed_layer_norm_detail

template <size_t Axis, IsFixedTensor TIn, IsFixedTensor TScale,
          IsFixedTensor TBias, IsFixedTensor TOut, typename TEp,
          typename PackedAxes, typename PadedNums>
void packed_layer_norm(const TIn &input, const TScale &scale, const TBias &bias,
                       TOut &&output, const TEp &epsilon, const bool &use_mean,
                       [[maybe_unused]] PackedAxes packedAxes,
                       [[maybe_unused]] PadedNums padednums) {
    static_assert(PackedAxes::rank() == 1, "currently not support 2d packing.");
    if constexpr (PackedAxes::rank() == 1) {
        static_assert(PackedAxes::at(0) >= Axis,
                      "currently only support pack within axis.");
        packed_layer_norm_detail::within_axis_pack_impl<Axis>(
            input, scale, bias, output, epsilon, use_mean);
    }
}
} // namespace nncase::ntt