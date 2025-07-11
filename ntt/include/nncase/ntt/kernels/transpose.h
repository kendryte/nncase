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
#include "../ukernels.h"
#include "nncase/ntt/shape.h"
#include <type_traits>

namespace nncase::ntt {
namespace transpose_detail {

template <Tensor TIn, class TOut, FixedDimensions TPerms> class transpose_impl {

  public:
    constexpr void operator()(const TIn &input, TOut &output, const TPerms &) {
        constexpr auto rank = TIn::rank();
        constexpr TPerms perm_const;
        constexpr auto pos_perms = positive_axes(perm_const, rank);
        ntt::apply(input.shape(), [&](auto index) {
            auto out_index = generate_shape<rank>(
                [&](auto i) { return index[pos_perms[i]]; });
            output(out_index) = input(index);
        });
    }
};
} // namespace transpose_detail

template <Tensor TIn, class TOut, FixedDimensions TPerms>
    requires(bool(TIn::rank() == std::decay_t<TOut>::rank()) &&
             bool(TIn::rank() == TPerms::rank()))
void transpose(
    const TIn &input, TOut &&output,
    const TPerms &perms = make_index_shape<TIn::rank()>().reverse()) {
    transpose_detail::transpose_impl<TIn, std::decay_t<TOut>, TPerms> impl;
    impl(input, output, perms);
}
} // namespace nncase::ntt
