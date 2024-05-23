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
#include "../loop.h"
#include "../primitive_ops.h"
#include "../tensor_ops.h"
#include "../utility.h"

namespace nncase::ntt {

namespace reduce_detail {

template <template <class T1, class T2> class Op, IsFixedTensor TIn,
          IsFixedTensor TOut, IsFixedDims Axes, IsFixedDims PackedAxes,
          IsFixedDims PadedNums>
void reduce_impl(const TIn &input, TOut &&output, Axes axes, PackedAxes,
                 PadedNums, [[maybe_unused]] bool keep_dims) {
    using TIElem = typename TIn::element_type;
    using TOElem = typename std::decay_t<TOut>::element_type;
    constexpr auto input_shape = typename TIn::shape_type{};
    constexpr auto input_strides = typename TIn::strides_type{};
    static_assert(
        is_same_seq(
            axes,
            slice_fixed_dims<Axes::rank(), input_shape.rank() - Axes::rank()>(
                make_index_sequence(input_shape))),
        "only support last axis for now!");
    constexpr auto output_shape = typename std::decay_t<TOut>::shape_type{};
    constexpr auto output_strides = typename std::decay_t<TOut>::strides_type{};

    constexpr size_t in_contigous_dim =
        contiguous_dims(input_shape, input_strides);
    constexpr size_t output_contiguous_dims =
        contiguous_dims(output_shape, output_strides);
    static_assert(in_contigous_dim != 0 || output_contiguous_dims != 0,
                  "only support contiguous for now!");

    constexpr auto domain = slice_fixed_dims<Axes::at(0)>(input_shape);
    constexpr auto strides = slice_fixed_dims<Axes::at(0)>(input_strides);
    constexpr auto ostrides = slice_fixed_dims<Axes::at(0)>(output_strides);

    constexpr size_t inner_size =
        slice_fixed_dims<Axes::rank(), input_shape.rank() - Axes::rank()>(
            input_shape)
            .length();
    constexpr bool UseVectorReduce =
        PackedAxes::rank() == 1 && PackedAxes::at(0) >= Axes::at(0);

    apply(domain, [&](auto index) {
        auto input_p = input.elements().data() + linear_offset(index, strides);
        auto output_p =
            output.elements().data() + linear_offset(index, ostrides);

        if constexpr (std::is_same_v<Op<TIElem, TIElem>,
                                     ntt::ops::mean<TIElem, TIElem>>) {
            TIElem finner_size = (TIElem)inner_size;
            if constexpr (UseVectorReduce) {
                finner_size =
                    finner_size * (TIElem)TIElem::shape_type::length();
            }

            TIElem mean = (TIElem)0;
            for (size_t i = 0; i < inner_size; i++)
                mean = mean + (input_p[i] / finner_size);
            if constexpr (UseVectorReduce) {
                output_p[0] = reduce_sum(mean);
            } else {
                output_p[0] = mean;
            }
        } else {
            auto iv =
                ntt::tensor_view<const TIElem, ntt::fixed_shape<inner_size>>(
                    std::span<const TIElem, inner_size>(input_p,
                                                        input_p + inner_size));
            auto ov = ops::reduce<Op, TIElem, decltype(iv)>()(iv);

            if constexpr (UseVectorReduce) {
                output_p[0] = ops::reduce<Op, TOElem, TIElem>()(ov);
            } else {
                output_p[0] = ov;
            }
        }
    });
}
} // namespace reduce_detail

template <template <class T1, class T2> class Op, IsFixedTensor TIn,
          IsFixedTensor TOut, IsFixedDims Axes, IsFixedDims PackedAxes,
          IsFixedDims PadedNums>
void reduce(const TIn &input, TOut &&output, Axes axes, PackedAxes packedAxes,
            PadedNums padedNums, bool keep_dims = true) noexcept {
    static_assert(PackedAxes::rank() < 2, "currently not support 2d packing.");

    static_assert(PadedNums::rank() == 0 ||
                      (PadedNums::rank() == 1 && PadedNums::at(0) == 0),
                  "not support padding");
    reduce_detail::reduce_impl<Op>(input, output, axes, packedAxes, padedNums,
                                   keep_dims);
}
} // namespace nncase::ntt
