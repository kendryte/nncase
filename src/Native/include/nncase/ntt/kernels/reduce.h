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
#include "../unrool.h"
#include "../utility.h"

namespace nncase::ntt {

namespace reduce_detail {

template <IsFixedDims Axes, IsFixedDims PackedAxes>
constexpr size_t unroll_arch() {
#if __riscv
    return 2;
#elif __x86_64__
    return (Axes::rank() >= 2 && PackedAxes::rank() == 0) ? 4 : 2;
#else
    return 1;
#endif
}

template <template <class T1, class T2> class Op, IsFixedTensor TIn,
          IsFixedTensor TOut, IsFixedDims Axes, IsFixedDims PackedAxes,
          IsFixedDims PadedNums>
void reduce_impl(const TIn &input, TOut &&output, Axes axes, PackedAxes,
                 PadedNums) {
    using TIElem = typename TIn::element_type;
    using TOElem = typename std::decay_t<TOut>::element_type;
    constexpr auto input_shape = typename TIn::shape_type{};
    constexpr auto input_strides = typename TIn::strides_type{};
    static_assert(is_same_seq(shift_fixed_dims<Axes::at(0)>(axes),
                              make_index_sequence(axes)),
                  "only support contiguous axis for now!");
    constexpr auto output_shape = typename std::decay_t<TOut>::shape_type{};
    constexpr auto output_strides = typename std::decay_t<TOut>::strides_type{};

    constexpr size_t in_contigous_dim =
        contiguous_dims(input_shape, input_strides);
    constexpr size_t output_contiguous_dims =
        contiguous_dims(output_shape, output_strides);
    static_assert(in_contigous_dim == input_shape.rank() &&
                      output_contiguous_dims == output_shape.rank(),
                  "only support contiguous for now!");

    constexpr auto domain = concat_fixed_dims(
        slice_fixed_dims<Axes::at(0)>(input_shape),
        slice_fixed_dims<input_shape.rank() - Axes::rank() - Axes::at(0),
                         Axes::at(0) + Axes::rank()>(input_shape));
    constexpr auto strides = concat_fixed_dims(
        slice_fixed_dims<Axes::at(0)>(input_strides),
        slice_fixed_dims<input_strides.rank() - Axes::rank() - Axes::at(0),
                         Axes::at(0) + Axes::rank()>(input_strides));

    [[maybe_unused]] constexpr auto ostrides = output_strides;
    constexpr auto rank =
        input_shape.rank() == output_shape.rank()
            ? output_strides.rank() - Axes::rank() - Axes::at(0)
            : 0;
    constexpr auto ostrides_keep_dims = concat_fixed_dims(
        slice_fixed_dims<Axes::at(0)>(output_strides),
        slice_fixed_dims<rank, Axes::at(0) + Axes::rank()>(output_strides));

    constexpr size_t inner_size =
        slice_fixed_dims<Axes::rank(), axes.at(0)>(input_shape).length();
    constexpr bool UseVectorReduce =
        PackedAxes::rank() == 1 && PackedAxes::at(0) >= Axes::at(0);

    constexpr size_t unroll_num = unroll_arch<Axes, PackedAxes>();

    constexpr auto input_stride = input_strides[Axes::at(Axes::rank() - 1)];
    apply(domain, [&](auto index) {
        auto input_p = input.elements().data() + linear_offset(index, strides);
        auto output_p = output.elements().data();
        if constexpr (input_shape.rank() == output_shape.rank()) {
            output_p += linear_offset(index, ostrides_keep_dims);
        } else {
            output_p += linear_offset(index, ostrides);
        }

        if constexpr (std::is_same_v<Op<TIElem, TIElem>,
                                     ntt::ops::mean<TIElem, TIElem>>) {
            TIElem sum;
            sum = loop_unrool<ntt::ops::add, TIElem, unroll_num, inner_size,
                              input_stride>(input_p);

            if constexpr (UseVectorReduce) {
                sum = sum / (inner_size * TIElem::shape_type::length());
                output_p[0] = reduce_sum(sum);
            } else {
                output_p[0] = sum / inner_size;
            }
        } else {
            TIElem ret;
            ret = loop_unrool<Op, TIElem, unroll_num, inner_size, input_stride>(
                input_p);

            if constexpr (UseVectorReduce) {
                output_p[0] = ops::reduce<Op, TOElem, TIElem>()(ret);
            } else {
                output_p[0] = ret;
            }
        }
    });
}
} // namespace reduce_detail

template <template <class T1, class T2> class Op, IsFixedTensor TIn,
          IsFixedTensor TOut, IsFixedDims Axes, IsFixedDims PackedAxes,
          IsFixedDims PadedNums>
void reduce(const TIn &input, TOut &&output, Axes axes, PackedAxes packedAxes,
            PadedNums padedNums) noexcept {
    static_assert(PackedAxes::rank() < 2, "currently not support 2d packing.");

    static_assert(PadedNums::rank() == 0 ||
                      (PadedNums::rank() == 1 && PadedNums::at(0) == 0),
                  "not support padding");
    AUTO_NTT_PROFILER
    reduce_detail::reduce_impl<Op>(input, output, axes, packedAxes, padedNums);
}
} // namespace nncase::ntt
