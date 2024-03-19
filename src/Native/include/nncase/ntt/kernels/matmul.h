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
#include "../shape_infer/reduce_axis.h"
#include "../utility.h"
#include "../vector_ops.h"
#include "binary.h"

namespace nncase::ntt {
namespace matmul_detail {

template <typename TElemtOut, typename TElemt>
constexpr inline TElemtOut dot(const TElemt &lp, const TElemt &rp) {
    constexpr mathops::mul<TElemt> mul;
    return mul(lp, rp);
}

template <typename TElemtOut, IsVector TElemt>
constexpr inline TElemtOut dot(const TElemt &lp, const TElemt &rp) {
    constexpr mathops::mul<TElemt> mul;
    constexpr vector_ops::reduce_sum<TElemt> rvsum;
    return rvsum(mul(lp, rp));
}

template <class TLhs, class TRhs, class TOut> struct matmul_impl;

/**
 * @brief fixed version
 */
template <IsFixedTensor TLhs, IsFixedTensor TRhs, IsFixedTensor TOut>
struct matmul_impl<TLhs, TRhs, TOut> {
    void operator()(const TLhs &lhs, const TRhs &rhs, TOut &output) {
        using TElemt = typename TLhs::element_type;
        using TElemtOut = typename TOut::element_type;
        constexpr auto lhs_cdim =
            contiguous_dims(TLhs::shape(), TLhs::strides());
        constexpr auto rhs_cdim =
            contiguous_dims(TRhs::shape(), TRhs::strides());
        constexpr auto out_cdim = contiguous_dims(
            std::decay_t<TOut>::shape(), std::decay_t<TOut>::strides());
        constexpr size_t lhs_rank = TLhs::shape().rank();
        constexpr size_t rhs_rank = TRhs::shape().rank();
        constexpr size_t out_rank = std::decay_t<TOut>::shape().rank();
        constexpr size_t M = TLhs::shape().at(lhs_rank - 2),
                         K = TLhs::shape().at(lhs_rank - 1),
                         N = TRhs::shape().at(rhs_rank - 1);
        constexpr mathops::add<TElemtOut> add;

        if constexpr (lhs_cdim >= 2 && rhs_cdim >= 2 && out_cdim >= 2) {
            constexpr auto domain = shape_infer::reduced_shape_by_axes(
                std::decay_t<TOut>::shape(),
                fixed_shape<out_rank - 1, out_rank - 2>{});

            auto lhs_index = ranked_shape<lhs_rank>{};
            auto rhs_index = ranked_shape<rhs_rank>{};
            apply(domain, [&](auto index) {
                loop<lhs_rank - 2>([&](auto i) {
                    lhs_index[i] = index[i + out_rank - lhs_rank];
                    if (lhs_index[i] >= TLhs::shape().at(i)) {
                        lhs_index[i] = TLhs::shape().at(i) - 1;
                    }
                });
                loop<rhs_rank - 2>([&](auto i) {
                    rhs_index[i] = index[i + out_rank - rhs_rank];
                    if (rhs_index[i] >= TRhs::shape().at(i)) {
                        rhs_index[i] = TRhs::shape().at(i) - 1;
                    }
                });

                auto lhs_p = lhs.buffer().data() +
                             linear_offset(lhs_index, lhs.strides());
                auto rhs_p = rhs.buffer().data() +
                             linear_offset(rhs_index, rhs.strides());
                auto output_p = output.buffer().data() +
                                linear_offset(index, output.strides());

                for (size_t m = 0; m < M; m++) {
                    for (size_t k = 0; k < 1; k++) {
                        for (size_t n = 0; n < N; n++) {
                            *(output_p + m * N + n) = dot<TElemtOut>(
                                *(lhs_p + m * K + k), *(rhs_p + k * N + n));
                        }
                    }
                    for (size_t k = 1; k < K; k++) {
                        for (size_t n = 0; n < N; n++) {
                            *(output_p + m * N + n) =
                                add(*(output_p + m * N + n),
                                    dot<TElemtOut>(*(lhs_p + m * K + k),
                                                   *(rhs_p + k * N + n)));
                        }
                    }
                }
            });
        } else {
            auto out_shape = output.shape();
            apply(out_shape, [&](auto index) {
                constexpr auto lrank = TLhs::shape_type::rank();
                constexpr auto rrank = TRhs::shape_type::rank();
                auto lhs_index = ranked_shape<lrank>{};
                auto rhs_index = ranked_shape<rrank>{};
                constexpr size_t lk = lhs_index.rank() - 1;
                constexpr size_t rk = rhs_index.rank() - 2;
                for (size_t i = 0; i < lk; i++) {
                    lhs_index[i] = index[i];
                    if (lhs_index[i] >= TLhs::shape().at(i)) {
                        lhs_index[i] = TLhs::shape().at(i) - 1;
                    }
                }
                for (size_t i = 0; i < rk; i++) {
                    rhs_index[i] = index[i];
                    if (rhs_index[i] >= TRhs::shape().at(i)) {
                        rhs_index[i] = TRhs::shape().at(i) - 1;
                    }
                }
                rhs_index[rk + 1] = index[rk + 1];
                TElemt acc = 0;
                for (lhs_index[lk] = 0; lhs_index[lk] < lhs.shape()[lk];
                     lhs_index[lk]++) {
                    rhs_index[rk] = lhs_index[lk];
                    TElemt val = mul(lhs(lhs_index), rhs(rhs_index));
                    acc = add(acc, val);
                }
                output(index) = acc;
            });
        }
    }
};
// matmul(const TLhs &lhs, const TRhs &rhs, TOut &&output) {}
} // namespace matmul_detail

template <class TLhs, class TRhs, class TOut>
void matmul(const TLhs &lhs, const TRhs &rhs, TOut &&output) {
    matmul_detail::matmul_impl<TLhs, TRhs, std::decay_t<TOut>> impl;
    impl(lhs, rhs, output);
}
} // namespace nncase::ntt
