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
template <FixedDimensions TPerms>
constexpr auto segments_cnt(const TPerms &) noexcept {
    constexpr auto rank = TPerms::rank();
    constexpr TPerms perm_const;
    return perm_const.template slice<0, rank - 1>().aggregate(
        1, [&](auto cnt_acc, auto perm, auto i) {
            if constexpr (perm != perm_const[i + 1_dim] - 1) {
                return cnt_acc + 1;
            } else {
                return cnt_acc;
            }
        });
}
} // namespace transpose_detail

template <Tensor TIn, class TOut, FixedDimensions TPerms>
    requires(bool(TIn::rank() == std::decay_t<TOut>::rank()) &&
             bool(TIn::rank() == TPerms::rank()))
void transpose(const TIn &input, TOut &&output,
               [[maybe_unused]] const TPerms &perms =
                   make_index_shape<TIn::rank()>().reverse()) {
    constexpr auto rank = TIn::rank();
    const auto conti_dims_input =
        contiguous_dims(input.shape(), input.strides());
    const auto conti_dims_output =
        contiguous_dims(output.shape(), output.strides());

    constexpr TPerms perm_const;
    constexpr auto segments = transpose_detail::segments_cnt(perm_const);

    if (segments <= 4 && conti_dims_input == rank &&
        conti_dims_output == rank) {
        u_transpose<TIn, std::decay_t<TOut>, TPerms, segments>(
            input, output, perms, std::make_index_sequence<segments>{});
    } else {
        constexpr auto pos_perms = positive_axes(perm_const, rank);

        ntt::apply(input.shape(), [&](auto index) {
            auto out_index = generate_shape<rank>(
                [&](auto i) { return index[pos_perms[i]]; });
            output(out_index) = input(index);
        });
    }
    // }
}
} // namespace nncase::ntt
