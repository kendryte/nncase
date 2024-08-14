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

namespace reduce_arg_detail {

template <template <class T1, class T2> class Op, size_t Axes,
          bool SelectLastIdx, bool KeepDims, IsFixedTensor TIn,
          IsFixedTensor TOut, IsFixedDims PackedAxes, IsFixedDims PadedNums>
void reduce_arg_impl(const TIn &input, TOut &&output, PackedAxes, PadedNums) {
    using TIElem = typename TIn::element_type;
    using TOElem = typename std::decay_t<TOut>::element_type;
    constexpr auto input_shape = typename TIn::shape_type{};
    constexpr auto output_shape = typename std::decay_t<TOut>::shape_type{};
    constexpr auto output_strides = typename std::decay_t<TOut>::strides_type{};

    static_assert(IsScalar<TOElem> && IsScalar<TIElem>,
                  "Only support scalar type for now");

    const float epsilon = 0.000001f;

    // use output tensor to store min/max values
    auto output_tensor = *reinterpret_cast<
        ntt::tensor<TIElem, typename std::decay_t<TOut>::shape_type> *>(
        output.elements().data());
    std::array<std::array<int64_t, 2>, output_shape[0] * output_strides[0]>
        out_map;
    if constexpr (std::is_same_v<Op<TIElem, TIElem>,
                                 ntt::ops::max<TIElem, TIElem>>) {
        apply(output_shape, [&](auto index) {
            output_tensor(index) = std::numeric_limits<TIElem>::lowest();
            auto out_offset = linear_offset(index, output_strides);
            out_map[out_offset][0] = -1;
            out_map[out_offset][1] = -1;
        });
    } else {
        apply(output_shape, [&](auto index) {
            output_tensor(index) = std::numeric_limits<TIElem>::max();
            auto out_offset = linear_offset(index, output_strides);
            out_map[out_offset][0] = -1;
            out_map[out_offset][1] = -1;
        });
    }

    constexpr size_t rank1 = input_shape.rank();
    constexpr size_t rank2 = input_shape.rank() - 1;

    // collect all min/max indices
    apply(input_shape, [&](auto index) {
        const auto src = input(index);
        size_t out_offset;
        if constexpr (KeepDims) {

            auto out_idx = get_reduced_offset<Axes, rank1>(index);
            out_offset = linear_offset(out_idx, output_strides);
        } else {
            auto out_idx = get_reduced_offset<Axes, rank2>(index);
            out_offset = linear_offset(out_idx, output_strides);
        }
        auto dst = output_tensor.elements().data() + out_offset;
        if constexpr (std::is_same_v<Op<TIElem, TIElem>,
                                     ntt::ops::max<TIElem, TIElem>>) {
            if (src > *dst) {
                out_map[out_offset][0] = index.at(Axes);
                *dst = src;
            } else {
                if constexpr (std::is_floating_point_v<TIElem>) {
                    if (std::fabs(src - *dst) < epsilon)
                        out_map[out_offset]
                               [out_map[out_offset][0] == -1 ? 0 : 1] =
                                   index.at(Axes);
                } else {
                    if (src == *dst)
                        out_map[out_offset]
                               [out_map[out_offset][0] == -1 ? 0 : 1] =
                                   index.at(Axes);
                }
            }
        } else {
            if (src < *dst) {
                out_map[out_offset][0] = (index.at(Axes));
                *dst = src;
            } else {
                if constexpr (std::is_floating_point_v<TIElem>) {
                    if (std::fabs(src - *dst) < epsilon)
                        out_map[out_offset]
                               [out_map[out_offset][0] == -1 ? 0 : 1] =
                                   index.at(Axes);
                } else {
                    if (src == *dst)
                        out_map[out_offset]
                               [out_map[out_offset][0] == -1 ? 0 : 1] =
                                   index.at(Axes);
                }
            }
        }
    });

    // update min/max idx
    apply(output_shape, [&](auto index) {
        auto out_offset = linear_offset(index, output_strides);
        if constexpr (SelectLastIdx) {
            output(index) =
                out_map[out_offset][out_map[out_offset][1] == -1 ? 0 : 1];
        } else {
            output(index) = out_map[out_offset][0];
        }
    });
}
} // namespace reduce_arg_detail

template <template <class T1, class T2> class Op, size_t Axes,
          bool SelectLastIdx, bool KeepDims, IsFixedTensor TIn,
          IsFixedTensor TOut, IsFixedDims PackedAxes, IsFixedDims PadedNums>
void reduce_arg(const TIn &input, TOut &&output, PackedAxes packedAxes,
                PadedNums padedNums) noexcept {
    static_assert(PackedAxes::rank() == 0, "currently not support packing.");
    static_assert(PadedNums::rank() == 0, "not support padding");

    reduce_arg_detail::reduce_arg_impl<Op, Axes, SelectLastIdx, KeepDims>(
        input, output, packedAxes, padedNums);
}
} // namespace nncase::ntt
