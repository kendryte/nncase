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
#include "../post_ops.h"
#include "../ukernels.h"
#include "detail/binary_like_impl.h"
#include <type_traits>

namespace nncase::ntt {
namespace detail {
template <Tensor TLhs, Tensor TRhs, Tensor TOut, template <class> class TPostOp>
class binary_impl
    : public binary_like_impl<binary_impl<TLhs, TRhs, TOut, TPostOp>, TLhs,
                              TRhs, TOut> {
  public:
    template <Tensor TBroadcastedLhs, Tensor TBroadcastedRhs, class TOp>
    void invoke_ukernel(const TBroadcastedLhs &lhs, const TBroadcastedRhs &rhs,
                        TOut &output, const TOp &op, bool is_broadcast) {

        auto lhs_conti_dims = contiguous_dims(lhs.shape(), lhs.strides());
        auto rhs_conti_dims = contiguous_dims(rhs.shape(), rhs.strides());
        auto output_conti_dims =
            contiguous_dims(output.shape(), output.strides());

        constexpr auto rank = TOut::rank();
        auto conti_dims = std::min(lhs_conti_dims, rhs_conti_dims);
        conti_dims = std::min(conti_dims, output_conti_dims);
        auto ref_shape = lhs.shape();

        auto lhs_u_strides = 1;
        auto rhs_u_strides = 1;
        constexpr auto zero_strides = make_zeros_strides<rank>();
        if (rhs.strides() == zero_strides) {
            conti_dims = std::min(lhs_conti_dims, output_conti_dims);
            ref_shape = lhs.shape();
            rhs_u_strides = 0;
        } else if (lhs.strides() == zero_strides) {
            conti_dims = std::min(rhs_conti_dims, output_conti_dims);
            ref_shape = rhs.shape();
            lhs_u_strides = 0;
        }

        auto apply_shape = generate_shape<rank>([&](auto i) {
            if (i > rank - conti_dims - 1)
                return (dim_t)1;
            else
                return (dim_t)ref_shape[i];
        });
        auto inner_shape = generate_shape<rank>([&](auto i) {
            if (i > rank - conti_dims - 1)
                return (dim_t)ref_shape[i];
            else
                return (dim_t)1_dim;
        });

        auto len = inner_shape.length();

        using TLhsElem = typename TLhs::element_type;
        using TRhsElem = typename TRhs::element_type;
        using TOutElem = typename TOut::element_type;
        TPostOp<TOutElem> post_op;

        // Todo: only support broadcast if one of the input is scalar
        // currently. Because this covers 90+% of the scenarios.
        if ((Vector<TLhsElem> && Vector<TRhsElem> && !is_broadcast) ||
            (Scalar<TLhsElem> && lhs_u_strides == 0) ||
            (Scalar<TRhsElem> && rhs_u_strides == 0)) {
            ntt::apply(apply_shape, [&](auto index) {
                const TLhsElem *NTT_RESTRICT addr_lhs = &lhs(index);
                const TRhsElem *NTT_RESTRICT addr_rhs = &rhs(index);
                TOutElem *NTT_RESTRICT addr_output_element = &output(index);
                ntt::u_binary<TOp, TPostOp, TLhsElem, TRhsElem, TOutElem>(
                    op, addr_lhs, lhs_u_strides, addr_rhs, rhs_u_strides,
                    addr_output_element, 1, len);
            });

        } else {
            const TLhsElem *NTT_RESTRICT lhs_p = lhs.elements().data();
            const TRhsElem *NTT_RESTRICT rhs_p = rhs.elements().data();
            TOutElem *NTT_RESTRICT output_p = output.elements().data();
            ntt::apply(
                output.shape(),
                [&](auto, auto lhs_offset, auto rhs_offset,
                    auto output_offset) {
                    auto value = op(lhs_p[lhs_offset], rhs_p[rhs_offset]);
                    output_p[output_offset] = post_op(value);
                },
                lhs.strides(), rhs.strides(), output.strides());
        }
    }
};
} // namespace detail

template <template <class T1, class T2> class TOp,
          template <class> class TPostOp = DefaultPostOp, Tensor TLhs,
          Tensor TRhs, class TOut>
void binary(const TLhs &lhs, const TRhs &rhs, TOut &&output) {
    const TOp<std::remove_cv_t<typename TLhs::element_type>,
              std::remove_cv_t<typename TRhs::element_type>>
        op;
    bool is_broadcast = lhs.shape() != rhs.shape();
    detail::binary_impl<TLhs, TRhs, std::decay_t<TOut>, TPostOp>()(
        lhs, rhs, output, op, is_broadcast);
}
} // namespace nncase::ntt
