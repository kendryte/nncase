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
#include "../shape_infer/pack.h"
#include "pack_element.h"

namespace nncase::ntt {
namespace detail {

template <class InShape, class OutShape, class InStrides, class OutStrides,
          size_t Axis>
class pack_impl;

template <size_t... InDims, size_t... OutDims, size_t... InStrides,
          size_t... OutStrides, size_t Axis>
class pack_impl<fixed_shape<InDims...>, fixed_shape<OutDims...>,
                fixed_strides<InStrides...>, fixed_strides<OutStrides...>,
                Axis> {
  public:
    template <class TIn, class TOut>
    constexpr void operator()(const TIn &input, TOut &&output) {
        using TElemIn = typename TIn::element_type;
        using TElemOut = typename std::decay_t<TOut>::element_type;
        constexpr size_t lanes = sizeof(TElemOut) / sizeof(TElemIn);
        apply(output.shape(), [&](auto out_index) {
            std::array<TElemIn, lanes> arr;
            loop<lanes>([&](auto i) {
                const auto src_index =
                    shape_infer::packed_index_by_shape<lanes, Axis, i>(
                        out_index);
                if (src_index[Axis] < input.shape()[Axis]) {
                    arr[i] = input(src_index);
                } else {
                    arr[i] = 0;
                }
            });
            output(out_index) = pack_elemt(arr);
        });
    }
};

} // namespace detail

template <size_t Axis, class TIn, class TOut>
void pack(const TIn &input, TOut &&output) noexcept {
    detail::pack_impl<typename TIn::shape_type,
                      typename std::decay_t<TOut>::shape_type,
                      typename TIn::strides_type,
                      typename std::decay_t<TOut>::strides_type, Axis>
        impl;
    impl(input, output);
}
} // namespace nncase::ntt
