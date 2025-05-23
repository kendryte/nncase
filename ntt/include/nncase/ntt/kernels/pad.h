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
#include "../padding.h"
#include "../utility.h"
#include "nncase/ntt/shape.h"

namespace nncase::ntt {
namespace pad_detail {
template <Tensor TIn, Paddings TPaddings, Scalar TElem, Tensor TOut>
void pad_impl(const TIn &input, TOut &output, const TPaddings &paddings,
              const TElem &pad_alue) {
    const auto input_shape = input.shape();
    constexpr auto rank = TIn::shape().rank();
    const auto output_shape = output.shape();
    dynamic_shape_t<rank> in_index{};
    apply(output_shape, [&](auto out_index) {
        bool dopad = false;
        loop<rank>([&](auto i) {
            in_index[i] = out_index[i] - paddings.at(i).before;
            if (in_index[i] < 0 || in_index[i] >= input_shape.at(i)) {
                dopad = true;
            }
        });
        if (dopad) {
            output(out_index) = pad_alue;
        } else {
            output(out_index) = input(in_index);
        }
    });
}
} // namespace pad_detail

/**
 * @brief pad
 *
 * @param input input tensor.
 * @param output output tensor.
 * @param pad_alue pad value.
 */
template <Tensor TIn, Paddings TPaddings,
          Scalar TElem = element_or_scalar_t<typename TIn::element_type>,
          class TOut>
    requires(TIn::rank() == TPaddings::rank())
void pad(const TIn &input, TOut &&output, const TPaddings &paddings,
         const TElem &pad_alue = {}) noexcept {
    pad_detail::pad_impl(input, paddings, pad_alue, output);
}
} // namespace nncase::ntt
