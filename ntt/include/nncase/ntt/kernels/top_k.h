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
#include "../ukernels.h"
#include "../utility.h"
#include "nncase/ntt/shape.h"
#include <cassert>
#include <iostream>
#include <stdio.h>

namespace nncase::ntt {

template <Tensor TOutProb, class TOutIndice, size_t Rank, size_t Axis>
void sort_descending(TOutProb &out_probs, TOutIndice &out_indices,
                     dynamic_shape_t<Rank> apply_index, int K) {
    for (int i = 0; i < K - 1; ++i) {
        auto cur_idx = generate_dims<Rank>([&](auto l) {
            if (l == Axis) {
                return (dim_t)i;
            } else {
                return (dim_t)apply_index[l];
            }
        });

        auto max_idx = cur_idx;

        for (int j = i + 1; j < K; ++j) {
            auto next_idx = generate_dims<Rank>([&](auto l) {
                if (l == Axis) {
                    return (dim_t)j;
                } else {
                    return (dim_t)apply_index[l];
                }
            });
            if (out_probs(next_idx) > out_probs(max_idx)) {
                max_idx = next_idx;
            }
        }

        if (max_idx != cur_idx) {
            auto tempValue = out_probs(cur_idx);
            out_probs(cur_idx) = out_probs(max_idx);
            out_probs(max_idx) = tempValue;

            auto tempIndex = out_indices(cur_idx);
            out_indices(cur_idx) = out_indices(max_idx);
            out_indices(max_idx) = tempIndex;
        }
    }
}

template <Tensor TOutProb, class TOutIndice, size_t Rank, size_t Axis>
void requeue_descending(TOutProb &out_probs, TOutIndice &out_indices,
                        float candidate_value,
                        dynamic_dims_t<Rank> candidate_index,
                        dynamic_dims_t<Rank> current_top_index, int64_t K) {
    auto next_candidate_value = out_probs(current_top_index);
    auto next_candidata_index = generate_dims<Rank>([&](auto j) {
        if (j == Axis) {
            return out_indices(current_top_index);
        } else {
            return candidate_index[j];
        }
    });

    out_probs(current_top_index) = candidate_value;
    out_indices(current_top_index) = candidate_index[Axis];

    for (int k = 0; k < K; k++) {
        auto top_index = generate_dims<Rank>([&](auto j) {
            if (j == Axis) {
                return (dim_t)k;
            } else {
                return (dim_t)current_top_index[j];
            }
        });

        if (next_candidate_value > out_probs(top_index)) {
            requeue_descending<TOutProb, TOutIndice, Rank, Axis>(
                out_probs, out_indices, next_candidate_value,
                next_candidata_index, top_index, K);
        }
    }
}

template <Tensor TInX, Tensor TInK, Tensor TOutProb, class TOutIndice,
          FixedDimension TAxis>
void top_k(const TInX &x, const TInK &k, TOutProb &out_probs,
           TOutIndice &out_indices, TAxis axis,
           [[maybe_unused]] int64_t largest, [[maybe_unused]] int64_t sorted) {

    constexpr auto Axis = dim_t(axis);
    constexpr auto rank = TInX::rank();
    auto K = k(0);
    auto apply_shape = generate_shape<rank>([&](auto i) {
        if (i == axis)
            return (dim_t)1;
        else
            return (dim_t)x.shape()[i];
    });
    auto inner_size = x.shape()[axis];
    ntt::apply(apply_shape, [&](auto apply_index) {
        for (int i = 0; i < K; i++) {
            auto top_index = generate_dims<rank>([&](auto j) {
                if (j == axis) {
                    return (dim_t)i;
                } else {
                    return (dim_t)apply_index[j];
                }
            });
            out_indices(top_index) = i;
            out_probs(top_index) = x(top_index);
        }

        for (int i = K; i < inner_size; i++) {
            auto candidate_index = generate_dims<rank>([&](auto j) {
                if (j == axis) {
                    return (dim_t)i;
                } else {
                    return (dim_t)apply_index[j];
                }
            });
            auto candidate_value = x(candidate_index);
            for (int k = 0; k < K; k++) {
                auto top_index = generate_dims<rank>([&](auto j) {
                    if (j == axis) {
                        return (dim_t)k;
                    } else {
                        return (dim_t)apply_index[j];
                    }
                });
                if (x(candidate_index) > out_probs(top_index)) {
                    requeue_descending<TOutProb, TOutIndice, rank(), Axis>(
                        out_probs, out_indices, candidate_value,
                        candidate_index, top_index, K);
                    break;
                }
            }
        }

        if (sorted) {
            if (largest) {
                sort_descending<TOutProb, TOutIndice, rank(), Axis>(
                    out_probs, out_indices, apply_index, K);
            }
        }
    });
}
} // namespace nncase::ntt
