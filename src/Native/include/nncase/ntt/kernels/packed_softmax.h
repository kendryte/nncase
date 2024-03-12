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
#include <algorithm>

namespace nncase::ntt {

namespace softmax_detail {

template <size_t Axis, size_t... Dims, size_t... Ints>
constexpr auto softmax_domain_impl(const fixed_shape<Dims...> shape,
                                   const std::index_sequence<Ints...>) {
    return fixed_shape<(Ints == Axis ? 1 : shape.at(Ints))...>{};
}
template <size_t Axis, size_t... Dims>
constexpr auto softmax_domain(const fixed_shape<Dims...> shape) {
    return softmax_domain_impl<Axis>(
        shape, std::make_index_sequence<sizeof...(Dims)>{});
}

template <size_t Axis, IsFixedTensor TIn, IsFixedTensor TOut>
void packed_on_axis_impl(const TIn &input, TOut &&output) {
    using TElem = TIn::element_type;
    constexpr auto input_shape = typename TIn::shape_type{};
    // constexpr auto input_strides = typename TIn::strides_type{};
    constexpr auto output_shape = typename std::decay_t<TOut>::shape_type{};
    // constexpr auto output_strides = typename
    // std::decay_t<TOut>::strides_type{};

    static_assert(is_same_seq(input_shape, output_shape),
                  "the input output shape not equal!");

    constexpr auto div_op = mathops::div<TElem>();
    constexpr auto exp_op = mathops::exp<TElem>();
    constexpr auto add_op = mathops::add<TElem>();
    constexpr auto sub_op = mathops::sub<TElem>();
    constexpr auto max_op = mathops::max<TElem>();
    constexpr auto vrmax_op = vector_ops::reduce_max<TElem>();
    constexpr auto vrsum_op = vector_ops::reduce_sum<TElem>();

    // TElem finner_size = inner_size;
    constexpr auto domain = softmax_domain<Axis>(input_shape);
    apply(domain, [&](auto index) {
        // reduce_max
        TElem max_value = input(index);
        for (index[Axis] = 0; index[Axis] < input_shape.at(Axis);
             index[Axis]++) {
            max_value = max_op(max_value, input(index));
        }
        max_value = vrmax_op(max_value);

        // (x - reduce_max) * beta
        for (index[Axis] = 0; index[Axis] < input_shape.at(Axis);
             index[Axis]++) {
            output(index) = sub_op(input(index), max_value);
        }

        // exp((x - reduce_max) * beta) and sum
        TElem sum = 0;
        for (index[Axis] = 0; index[Axis] < input_shape.at(Axis);
             index[Axis]++) {
            output(index) = exp_op(output(index));
            sum = add_op(output(index), sum);
        }
        sum = vrsum_op(sum);

        // div
        for (index[Axis] = 0; index[Axis] < input_shape.at(Axis);
             index[Axis]++) {
            output(index) = div_op(output(index), sum);
        }
    });
}

template <size_t Axis, IsFixedTensor TIn, IsFixedTensor TOut,
          typename PackedAxes>
void packed_softmax_1d(const TIn &input, TOut &&output, PackedAxes) {
    static_assert(Axis == PackedAxes::at(0),
                  "currently not support packed on reduce axis");
    packed_on_axis_impl<Axis>(input, output);
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
 * @param padednums packed axes's paded nums
 */
template <size_t Axis, IsFixedTensor TIn, IsFixedTensor TOut,
          typename PackedAxes, typename PadedNums>
void packed_softmax(const TIn &input, TOut &&output,
                    [[maybe_unused]] PackedAxes packedAxes,
                    [[maybe_unused]] PadedNums padednums) noexcept {
    static_assert(PackedAxes::rank() == 1, "currently not support 2d pack");
    static_assert(PadedNums::at(0) == 0, "currently not support pad");
    softmax_detail::packed_softmax_1d<Axis>(input, output, packedAxes);
}
} // namespace nncase::ntt
