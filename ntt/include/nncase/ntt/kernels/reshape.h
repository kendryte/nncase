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
#include "../tensor_traits.h"
#include "../ukernels.h"
#include "../utility.h"
#include <type_traits>

namespace nncase::ntt {
namespace detail {
template <class TIn, class TOut> class reshape_impl {
  public:
    constexpr void operator()(const TIn &input, TOut &output) {
        const size_t size = input.size();
        const auto in_shape_strides = default_strides(input.shape());
        const auto out_shape_strides = default_strides(output.shape());
        for (size_t i = 0; i < size; i++) {
            auto in_index = unravel_index(i, in_shape_strides);
            auto out_index = unravel_index(i, out_shape_strides);
            output(out_index) = input(in_index);
        }
    }
};
} // namespace detail

template <class TIn, class TOut> void reshape(const TIn &input, TOut &&output) {
    detail::reshape_impl<std::decay_t<TIn>, std::decay_t<TOut>>()(input,
                                                                  output);
}
} // namespace nncase::ntt
