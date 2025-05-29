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
constexpr auto segments_cnt(const TPerms &perms) noexcept {
    return perms.template slice<1>.aggregate(
        1_dim, [&](auto cnt_acc, auto perm, auto i) {
            if constexpr (perm != perms[i - 1_dim] + 1) {
                return cnt_acc + 1;
            } else {
                return cnt_acc;
            }
        });
}
} // namespace transpose_detail

template <Tensor TIn, class TOut, FixedDimensions TPerms>
    requires(TIn::rank() == std::decay_t<TOut>::rank() &&
             TIn::rank() == TPerms::rank())
void transpose(
    const TIn &input, TOut &&output,
    const TPerms &perms = make_index_shape<TIn::rank()>().reverse()) {
    constexpr auto rank = TIn::rank();
    const auto pos_perms = positive_axes(perms, rank);
    // constexpr auto segs_cnt = transpose_detail::segments_cnt(perms);
    // const auto cdims_input = contiguous_dims(input.shape(), input.strides());
    // const auto cdims_output = contiguous_dims(output.shape(),
    // output.strides());

    // if constexpr (cdims_input == rank && cdims_output == rank &&
    //               segs_cnt <= 4) {
    //     ntt::u_transpose(input, output, perms,
    //                      std::make_index_sequence<segs_cnt>{});
    // } else {
    ntt::apply(input.shape(), [&](auto index) {
        auto out_index =
            generate_shape<rank>([&](auto i) { return index[pos_perms[i]]; });
        output(out_index) = input(index);
    });
    // }
}
} // namespace nncase::ntt
