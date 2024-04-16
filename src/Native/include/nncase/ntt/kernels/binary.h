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
#include "../shape_infer/binary.h"
#include "../shape_infer/reduce.h"
#include "../tensor_traits.h"
#include <type_traits>

namespace nncase::ntt {
namespace detail {
template <class TLhs, class TRhs, class TOut> class binary_impl {
  public:
    template <class Op>
    constexpr void operator()(Op &op, const TLhs &lhs, const TRhs &rhs,
                              TOut &output) {
        auto out_shape =
            shape_infer::binary_output_shape(lhs.shape(), rhs.shape());

        apply(out_shape, [&](auto index) {
            const auto lhs_index =
                shape_infer::reduced_index_by_shape(index, lhs.shape());
            const auto rhs_index =
                shape_infer::reduced_index_by_shape(index, rhs.shape());
            output(index) = op(lhs(lhs_index), rhs(rhs_index));
        });
    }
};

template <IsFixedTensor TLhs, IsFixedTensor TRhs, IsFixedTensor TOut>
class binary_impl<TLhs, TRhs, TOut> {
  public:
    template <class Op>
    constexpr void operator()(Op &op, const TLhs &lhs, const TRhs &rhs,
                              TOut &output) {
        auto out_shape =
            shape_infer::binary_output_shape(lhs.shape(), rhs.shape());

        apply(out_shape, [&](auto index) {
            const auto lhs_index =
                shape_infer::reduced_index_by_shape(index, lhs.shape());
            const auto rhs_index =
                shape_infer::reduced_index_by_shape(index, rhs.shape());
            output(index) = op(lhs(lhs_index), rhs(rhs_index));
        });
    }
};
} // namespace detail

template <template <class T1, class T2> class Op, class TLhs, class TRhs,
          class TOut>
void binary(const TLhs &lhs, const TRhs &rhs, TOut &output) {
    Op<typename TLhs::element_type, typename TRhs::element_type> op;
    detail::binary_impl<std::decay_t<TLhs>, std::decay_t<TRhs>,
                        std::decay_t<TOut>>()(op, lhs, rhs, output);
}
} // namespace nncase::ntt
