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

namespace nncase::ntt {
namespace pad_detail {
template <Tensor TIn, Tensor TOut, Paddings TPaddings, ScalarOrVector TElem>
void pad_impl(const TIn &input, TOut &output, const TPaddings &paddings,
              const TElem &pad_alue) {
    constexpr auto rank = TIn::rank();
    ntt::apply(output.shape(), [&](auto out_index) {
        bool dopad = false;
        const auto in_index = generate_shape<rank>([&](auto i) {
            auto in_dim = out_index[i] - paddings[i].before;
            if (in_dim < 0 || in_dim >= input.shape()[i]) {
                dopad = true;
            }
            return in_dim;
        });
        if (dopad) {
            output(out_index) = (typename TIn::element_type)pad_alue;
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
template <Tensor TIn, class TOut, Paddings TPaddings,
          ScalarOrVector TElem = typename TIn::element_type>
    requires(bool(TIn::rank() == TPaddings::rank()))
void pad(const TIn &input, TOut &&output, const TPaddings &paddings,
         const TElem &pad_alue = {}) noexcept {
    pad_detail::pad_impl(input, output, paddings, pad_alue);
}
} // namespace nncase::ntt
