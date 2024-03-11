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
#include "../vector_ops.h"
#include "binary.h"
#include "unary.h"

namespace nncase::ntt {

template <typename T>
concept IsFixedTensor = requires(T t) {
    typename std::decay_t<T>::shape_type;
    is_fixed_dims_v<typename std::decay_t<T>::shape_type>;
};

namespace packed_layer_norm_detail {

template <size_t Axis, IsFixedTensor TIn, IsFixedTensor TScale,
          IsFixedTensor TBias, IsFixedTensor TOut, typename TEp>
void within_axis_pack_impl(const TIn &input, const TScale &scale,
                           const TBias &bias, TOut &&output, const TEp &epsilon,
                           [[maybe_unused]] const bool &use_mean) {
    using TElem = TIn::element_type;
    using TScalar = typename TIn::element_type::element_type;
    constexpr size_t in_contigous_dim =
        contiguous_dims(input.shape(), input.strides());
    constexpr size_t scale_contiguous_dims =
        contiguous_dims(scale.shape(), scale.strides());
    constexpr size_t bias_contiguous_dims =
        contiguous_dims(bias.shape(), bias.strides());
    constexpr size_t output_contiguous_dims =
        contiguous_dims(output.shape(), output.strides());
    static_assert(in_contigous_dim != 0 || scale_contiguous_dims != 0 ||
                      bias_contiguous_dims != 0 || output_contiguous_dims != 0,
                  "currently not support no contiguous!");
    static_assert(is_same_seq(input.shape(), output.shape()),
                  "shape not match");
    static_assert(is_same_seq(input.strides(), output.strides()),
                  "strides not match");
    constexpr auto domain = slice<Axis>(input.shape());
    constexpr auto strides = slice<Axis>(input.strides());
    constexpr size_t inner_size =
        slice<input.shape().rank() - Axis, Axis>(input.shape()).length();
    constexpr auto sqrt_op = mathops::sqrt<TElem>();
    constexpr auto div_op = mathops::div<TElem>();
    constexpr auto sub_op = mathops::sub<TElem>();
    constexpr auto vsum_op = vector_ops::sum<TElem>();

    TElem finner_size = inner_size * TElem::shape_type::length();

    apply(domain, [&](auto index) {
        auto input_p = input.buffer().data() + linear_offset(index, strides);
        auto scale_p = scale.buffer().data();
        auto bias_p = bias.buffer().data();
        auto output_p = output.buffer().data() + linear_offset(index, strides);

        // compute mean
        TElem mean1tmp = 0;
        TElem mean1 = 0;
        if (use_mean) {
            for (size_t i = 0; i < inner_size; i++)
                mean1tmp = mean1tmp + div_op(input_p[i], finner_size);
            mean1 = vsum_op(mean1tmp);
        }

        std::array<TElem, inner_size> sub;
        for (auto i = 0; i < inner_size; i++)
            sub[i] = sub_op(input_p[i], mean1);

        std::array<TElem, inner_size> pow;
        for (auto i = 0; i < inner_size; i++)
            pow[i] = sub[i] * sub[i];

        TElem mean2temp = 0;
        for (auto i = 0; i < inner_size; i++)
            mean2temp = mean2temp + (pow[i] / finner_size);
        TElem mean2 = vsum_op(mean2temp);

        TElem add = mean2 + epsilon;
        TElem sqrt = sqrt_op(add);

        std::array<TElem, inner_size> norm;
        for (auto i = 0; i < inner_size; i++)
            norm[i] = sub[i] / sqrt;

        for (auto i = 0; i < inner_size; i++)
            output_p[i] = norm[i] * scale_p[i] + bias_p[i];
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
