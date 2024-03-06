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
#include "../shape_infer/unpack.h"
#include "unpack_element.h"

namespace nncase::ntt {
namespace detail {

template <class InShape, class OutShape, class InStrides, class OutStrides,
          size_t Axis>
class unpack_impl;

template <size_t... InDims, size_t... OutDims, size_t... InStrides,
          size_t... OutStrides, size_t Axis>
class unpack_impl<fixed_shape<InDims...>, fixed_shape<OutDims...>,
                  fixed_strides<InStrides...>, fixed_strides<OutStrides...>,
                  Axis> {
  public:
    template <class TIn, class TOut>
    constexpr void operator()(const TIn &input, TOut &&output) {
        using TVec = typename TIn::element_type;
        using TScalar = typename std::decay_t<TOut>::element_type;
        constexpr size_t lanes = sizeof(TVec) / sizeof(TScalar);
        apply(input.shape(), [&](auto input_index) {
            std::array<TScalar, lanes> arr;
            const auto output_index =
                shape_infer::unpacked_index_by_shape<lanes, Axis>(input_index);
            const TVec vec = input(input_index);
            const size_t output_offset =
                linear_offset(output_index, output.strides());
            auto iter = output.buffer().begin() + output_offset;
            unpack_elemt(arr, vec);
            loop<lanes>([&](auto i) {
                if (output_index[Axis] + i <= output.shape()[Axis]) {
                    *(iter + i * output.strides()[Axis]) = arr[i];
                }
            });
        });
    }
};

} // namespace detail

template <size_t Axis, class TIn, class TOut>
void unpack(const TIn &input, TOut &&output) noexcept {
    detail::unpack_impl<typename TIn::shape_type,
                        typename std::decay_t<TOut>::shape_type,
                        typename TIn::strides_type,
                        typename std::decay_t<TOut>::strides_type, Axis>
        impl;
    impl(input, output);
}
} // namespace nncase::ntt
