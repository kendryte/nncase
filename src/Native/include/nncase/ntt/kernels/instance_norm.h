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
#include "../tensor_ops.h"
#include "../utility.h"
#include "binary.h"
#include "unary.h"

namespace nncase::ntt {

namespace instance_norm_detail {

template <IsFixedTensor TIn, IsFixedTensor TScale, IsFixedTensor TBias,
          IsFixedTensor TOut, typename TEp, IsFixedDims PackedAxes,
          IsFixedDims PadedNums>
void impl(const TIn &input, const TScale &scale, const TBias &bias,
          TOut &&output, const TEp &epsilon, PackedAxes, PadedNums) {
    using TElem = typename TIn::element_type;
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
    constexpr size_t Axis = 2;
    constexpr auto domain = slice_fixed_dims<Axis>(input_shape);
    constexpr auto strides = slice_fixed_dims<Axis>(input_strides);

    constexpr size_t inner_size =
        slice_fixed_dims<input_shape.rank() - Axis, Axis>(input_shape).length();
    constexpr bool UseVectorReduce = PackedAxes::rank() == 1;

    TElem finner_size = (TElem)inner_size;
    if constexpr (UseVectorReduce) {
        finner_size = finner_size * (TElem)TElem::shape_type::length();
    }
    // remove pad nums, NOTE after mul elem size
    // finner_size = sub_op(finner_size, paded_inner_size);

    apply(domain, [&](auto index) {
        const auto input_p =
            input.elements().data() + linear_offset(index, strides);
        auto output_p =
            output.elements().data() + linear_offset(index, strides);

        // compute mean
        TElem mean1 = (TElem)0;
        for (size_t i = 0; i < inner_size; i++)
            mean1 = mean1 + (input_p[i] / finner_size);
        if constexpr (UseVectorReduce) {
            mean1 = (TElem)reduce_sum(mean1);
        }

        std::array<TElem, inner_size> sub;
        for (auto i = 0; i < inner_size; i++)
            sub[i] = input_p[i] - mean1;

        std::array<TElem, inner_size> pow;
        for (auto i = 0; i < inner_size; i++)
            pow[i] = sub[i] * sub[i];

        TElem mean2 = (TElem)0;
        for (auto i = 0; i < inner_size; i++)
            mean2 = mean2 + (pow[i] / finner_size);
        if constexpr (UseVectorReduce) {
            mean2 = (TElem)reduce_sum(mean2);
        }

        TElem add = mean2 + epsilon;
        TElem sqrt = ntt::sqrt(add);

        std::array<TElem, inner_size> norm;
        for (auto i = 0; i < inner_size; i++)
            norm[i] = sub[i] / sqrt;

        for (auto i = 0; i < inner_size; i++)
            output_p[i] = (norm[i] * (TElem)scale(ranked_shape<1>{index[1]})) +
                          (TElem)bias(ranked_shape<1>{index[1]});
    });
}
} // namespace instance_norm_detail

template <typename TIn, typename TScale, typename TBias, typename TOut,
          typename TEp, IsFixedDims PackedAxes, IsFixedDims PadedNums>
void instance_norm(const TIn &input, const TScale &scale, const TBias &bias,
                   TOut &&output, const TEp &epsilon, PackedAxes packedAxes,
                   PadedNums padedNums) {
    static_assert(PackedAxes::rank() < 2, "currently not support 2d packing.");
    if constexpr (PackedAxes::rank() <= 1) {
        static_assert(PadedNums::rank() == 0 ||
                          (PadedNums::rank() == 1 && PadedNums::at(0) == 0),
                      "not support padding");
        instance_norm_detail::impl(input, scale, bias, output, epsilon,
                                   packedAxes, padedNums);
    }
}
} // namespace nncase::ntt
