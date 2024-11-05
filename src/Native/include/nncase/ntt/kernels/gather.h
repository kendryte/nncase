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
#include "../tensor_ops.h"
#include "../ukernels.h"
#include "../utility.h"
#include <cstddef>
#include <iostream>
namespace nncase::ntt {

namespace detail {

template <class Shape, class InStrides, class OutStrides> class gather_impl;

template <size_t... Dims, size_t... InStrides, size_t... OutStrides>
class gather_impl<fixed_shape<Dims...>, fixed_strides<InStrides...>,
                  fixed_strides<OutStrides...>> {
  public:
    template <size_t Axis, typename TA, typename TB, typename TC>
    constexpr void operator()(const TA &input, const TB &indices, TC &&output) {
        constexpr auto rank = TA::shape_type::rank();
        using element_type = element_or_scalar_t<TA>;
        using slice_type = element_or_scalar_t<TB>;
        constexpr auto element_size = sizeof(element_type);

        constexpr size_t indices_len = TB::size();

        Segment segments[indices_len];
        size_t count = find_continuous_segments(
            (const slice_type *)(indices.elements().data()), indices_len,
            segments);

        auto domain_before_axis = slice_fixed_dims<Axis>(input.shape());
        constexpr auto domain_after_axis =
            slice_fixed_dims<rank - Axis - 1, Axis + 1>(TA::shape());

        auto addr_output_byte =
            reinterpret_cast<unsigned char *>(output.buffer().data());
        auto addr_output_element = output.buffer().data();

        auto input_conti_dims = contiguous_dims(input.shape(), input.strides());

        constexpr auto indices_rank = TB::shape_type::rank();
        constexpr auto out_shape = std::decay_t<TC>::shape();
        ranked_shape<rank> in_index;
        ranked_shape<indices_rank> indices_index;
        ranked_shape<rank> src_index;
        for (size_t i = 0; i < rank; i++) {
            src_index[i] = 0;
        }

        if (input_conti_dims == rank && count != indices_len) {
            apply(domain_before_axis, [&](auto index) {
                for (size_t i = 0; i < count; i++) {
                    auto seq = segments[i];
                    for (size_t j = 0; j < Axis; j++) {
                        src_index[j] = index[j];
                    }
                    src_index[Axis] = indices.elements()[seq.start];
                    auto len =
                        seq.length * domain_after_axis.length() * element_size;
                    std::memcpy(addr_output_byte, &(input(src_index)), len);
                    addr_output_byte += len;
                }
            });
        } else if (input_conti_dims == rank) {
            apply(domain_before_axis, [&](auto index) {
                for (size_t i = 0; i < TB::size(); i++) {

                    for (size_t j = 0; j < Axis; j++) {
                        src_index[j] = index[j];
                    }
                    src_index[Axis] = indices.elements()[i];
                    auto addr_input = reinterpret_cast<const element_type *>(
                        &(input(src_index)));
                    constexpr auto len = domain_after_axis.length();

                    ntt::u_memcpy<element_type>(addr_input, 1,
                                                addr_output_element, 1, len);
                    addr_output_element += len;
                }
            });
        } else {
            apply(out_shape, [&](auto out_index) {
                // in_index[:axis] = out_index[:axis]
                loop<Axis>([&](auto i) { in_index[i] = out_index[i]; });

                // in_index[axis] = indices(indices_index)
                loop<indices_rank>(
                    [&](auto i) { indices_index[i] = out_index[i + Axis]; });
                in_index[Axis] = indices(indices_index);

                // in_index[axis:] = out_index[axis:]
                loop<rank - (Axis + 1)>([&](auto i) {
                    in_index[Axis + 1 + i] = out_index[Axis + indices_rank + i];
                });
                output(out_index) = input(in_index);
            });
        }
    }

  private:
    struct Segment {
        size_t start;
        size_t length;
    };

    template <typename T>
    size_t find_continuous_segments(const T *arr, size_t arrSize,
                                    Segment *segments) {
        if (arrSize == 0)
            return 0;

        size_t segment_count = 0;
        size_t start = 0;
        size_t length = 1;

        for (size_t i = 1; i < arrSize; ++i) {
            if (arr[i] == arr[i - 1] + 1) {
                ++length;
            } else {
                segments[segment_count].start = start;
                segments[segment_count].length = length;
                ++segment_count;
                start = i;
                length = 1;
            }
        }

        segments[segment_count].start = start;
        segments[segment_count].length = length;
        ++segment_count;

        return segment_count;
    }
};

template <size_t Rank, class InStrides, class OutStrides>
class gather_impl<ranked_shape<Rank>, InStrides, OutStrides> {
  public:
    template <size_t Axis, typename TA, typename TB, typename TC>
    constexpr void operator()(const TA &input, const TB &indices, TC &&output) {
        ranked_shape<Rank> in_index;
        constexpr auto indices_rank = TB::shape_type::rank();
        ranked_shape<indices_rank> indices_index;
        apply(output.shape(), [&](auto out_index) {
            // in_index[:axis] = out_index[:axis]
            loop<Axis>([&](auto i) { in_index[i] = out_index[i]; });

            // in_index[axis] = indices(indices_index)
            loop<indices_rank>(
                [&](auto i) { indices_index[i] = out_index[i + Axis]; });
            in_index[Axis] = indices(indices_index);

            // in_index[axis:] = out_index[axis:]
            loop<Rank - (Axis + 1)>([&](auto i) {
                in_index[Axis + 1 + i] = out_index[Axis + indices_rank + i];
            });
            output(out_index) = input(in_index);
        });
    }
};

} // namespace detail

template <size_t Axis, typename TA, typename TB, typename TC>
void gather(const TA &input, const TB &indices, TC &&output) noexcept {
    detail::gather_impl<typename TA::shape_type, typename TA::strides_type,
                        typename std::decay_t<TC>::strides_type>
        impl;
    impl.template operator()<Axis>(input, indices, output);
}

} // namespace nncase::ntt
