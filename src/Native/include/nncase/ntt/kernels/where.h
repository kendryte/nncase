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
#include "../utility.h"

namespace nncase::ntt {

namespace where_detail {

template <IsFixedTensor TCond, IsFixedTensor TX, IsFixedTensor TY,
          IsFixedTensor TOut>
void where_impl(const TCond &cond, const TX &x, const TY &y,
                TOut &&output) noexcept {
    // constexpr auto rank = TOut::shape_type::rank();
    // constexpr auto input_strides = TIn::strides();
    // constexpr auto output_strides = std::decay_t<TOut>::strides();

    static_assert(IsScalar<typename std::decay_t<TOut>::element_type>,
                  "Only support scalar type for now");

    apply(output.shape(), [&](auto index) {
        const auto cond_index = get_reduced_offset(index, cond.shape());
        const auto x_index = get_reduced_offset(index, x.shape());
        const auto y_index = get_reduced_offset(index, y.shape());

        const auto a = cond(cond_index);
        const auto b = x(x_index);
        const auto c = y(y_index);

        output(index) = a ? b : c;
    });
}
} // namespace where_detail

template <typename TCond, typename TX, typename TY, typename TOut>
void where(const TCond &cond, const TX &x, const TY &y,
           TOut &&output) noexcept {
    where_detail::where_impl(cond, x, y, output);
}
} // namespace nncase::ntt
