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

namespace nncase::ntt {
namespace copy_detail {
template <typename TA, typename TB> struct copy_impl;
template <IsFixedTensor TA, IsFixedTensor TB> struct copy_impl<TA, TB> {
    constexpr void operator()(const TA &input, TB &output) {
        constexpr auto input_shape = TA::shape();
        constexpr auto input_strides = TA::strides();

        constexpr auto output_shape = std::decay_t<TB>::shape();
        constexpr auto output_strides = std::decay_t<TB>::strides();

        constexpr auto cdim_input = contiguous_dims(input_shape, input_strides);
        constexpr auto cdim_output =
            contiguous_dims(output_shape, output_strides);

        if constexpr (cdim_input == cdim_output &&
                      cdim_input == input_shape.rank() &&
                      cdim_output == output_shape.rank()) {
            auto out_buffer = output.buffer();
            memcpy(out_buffer.data(), input.buffer().data(),
                   out_buffer.size_bytes());
        } else {
            apply(input_shape,
                  [&](auto index) { output(index) = input(index); });
        }
    }
};
} // namespace copy_detail

template <class TA, class TB>
void tensor_copy(const TA &input, TB &&output) noexcept {
    copy_detail::copy_impl<TA, TB> impl;
    impl(input, output);
}
} // namespace nncase::ntt
