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
#include "../loop.h"
#include <cstddef>
#include <type_traits>

namespace nncase::ntt {
namespace u_transpose_detail {

template <Tensor TIn, class TOut, FixedDimensions TPerms>
class u_transpose_impl {

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

#define DEFINE_U_TRANSPOSE_IMPL(PERM0, PERM1, PERM2, PERM3, ACCESS_EXPR)       \
    template <Tensor TIn, class TOut>                                          \
    class u_transpose_impl<TIn, TOut,                                          \
                           fixed_shape_t<PERM0, PERM1, PERM2, PERM3>> {        \
      public:                                                                  \
        constexpr void                                                         \
        operator()(const TIn &input, TOut &output,                             \
                   const fixed_shape_t<PERM0, PERM1, PERM2, PERM3> &) {        \
            for (auto i = 0; i < input.shape()[0]; ++i)                        \
                for (auto j = 0; j < input.shape()[1]; ++j)                    \
                    for (auto k = 0; k < input.shape()[2]; ++k)                \
                        for (auto l = 0; l < input.shape()[3]; ++l)            \
                            output ACCESS_EXPR = input(i, j, k, l);            \
        }                                                                      \
    }

DEFINE_U_TRANSPOSE_IMPL(0, 1, 2, 3, (i, j, k, l));
DEFINE_U_TRANSPOSE_IMPL(0, 1, 3, 2, (i, j, l, k));
DEFINE_U_TRANSPOSE_IMPL(0, 2, 1, 3, (i, k, j, l));
DEFINE_U_TRANSPOSE_IMPL(0, 2, 3, 1, (i, k, l, j));
DEFINE_U_TRANSPOSE_IMPL(0, 3, 1, 2, (i, l, j, k));
DEFINE_U_TRANSPOSE_IMPL(0, 3, 2, 1, (i, l, k, j));

DEFINE_U_TRANSPOSE_IMPL(1, 0, 2, 3, (j, i, k, l));
DEFINE_U_TRANSPOSE_IMPL(1, 0, 3, 2, (j, i, l, k));
DEFINE_U_TRANSPOSE_IMPL(1, 2, 0, 3, (j, k, i, l));
DEFINE_U_TRANSPOSE_IMPL(1, 2, 3, 0, (j, k, l, i));
DEFINE_U_TRANSPOSE_IMPL(1, 3, 0, 2, (j, l, i, k));
DEFINE_U_TRANSPOSE_IMPL(1, 3, 2, 0, (j, l, k, i));

DEFINE_U_TRANSPOSE_IMPL(2, 0, 1, 3, (k, i, j, l));
DEFINE_U_TRANSPOSE_IMPL(2, 0, 3, 1, (k, i, l, j));
DEFINE_U_TRANSPOSE_IMPL(2, 1, 0, 3, (k, j, i, l));
DEFINE_U_TRANSPOSE_IMPL(2, 1, 3, 0, (k, j, l, i));
DEFINE_U_TRANSPOSE_IMPL(2, 3, 0, 1, (k, l, i, j));
DEFINE_U_TRANSPOSE_IMPL(2, 3, 1, 0, (k, l, j, i));

DEFINE_U_TRANSPOSE_IMPL(3, 0, 1, 2, (l, i, j, k));
DEFINE_U_TRANSPOSE_IMPL(3, 0, 2, 1, (l, i, k, j));
DEFINE_U_TRANSPOSE_IMPL(3, 1, 0, 2, (l, j, i, k));
DEFINE_U_TRANSPOSE_IMPL(3, 1, 2, 0, (l, j, k, i));
DEFINE_U_TRANSPOSE_IMPL(3, 2, 0, 1, (l, k, i, j));
DEFINE_U_TRANSPOSE_IMPL(3, 2, 1, 0, (l, k, j, i));

} // namespace u_transpose_detail

template <Tensor TIn, class TOut, FixedDimensions TPerms>
    requires(bool(TIn::rank() == std::decay_t<TOut>::rank()) &&
             bool(TIn::rank() == TPerms::rank()))
void u_transpose(
    const TIn &input, TOut &&output,
    const TPerms &perms = make_index_shape<TIn::rank()>().reverse()) {
    u_transpose_detail::u_transpose_impl<TIn, std::decay_t<TOut>, TPerms> impl;
    impl(input, output, perms);
}

} // namespace nncase::ntt
