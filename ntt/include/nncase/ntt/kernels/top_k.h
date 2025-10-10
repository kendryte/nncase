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

template <size_t Rank, size_t Axis>
void sort_descending(float *out_probs, int64_t *out_indices,
                     int64_t out_probs_stride, int64_t out_indices_stride,
                     int K) {
    for (int cur_idx = 0; cur_idx < K - 1; ++cur_idx) {
        auto max_idx = cur_idx;

        for (int next_idx = cur_idx + 1; next_idx < K; ++next_idx) {
            if (out_probs[next_idx * out_probs_stride] >
                out_probs[max_idx * out_probs_stride]) {
                max_idx = next_idx;
            }
        }

        if (max_idx != cur_idx) {
            auto tempValue = out_probs[cur_idx * out_probs_stride];
            out_probs[cur_idx * out_probs_stride] =
                out_probs[max_idx * out_probs_stride];
            out_probs[max_idx * out_probs_stride] = tempValue;

            auto tempIndex = out_indices[cur_idx * out_indices_stride];
            out_indices[cur_idx * out_indices_stride] =
                out_indices[max_idx * out_indices_stride];
            out_indices[max_idx * out_indices_stride] = tempIndex;
        }
    }
}

template <size_t Rank, size_t Axis>
void sort_ascending(float *out_probs, int64_t *out_indices,
                    int64_t out_probs_stride, int64_t out_indices_stride,
                    int K) {
    for (int cur_idx = 0; cur_idx < K - 1; ++cur_idx) {
        auto min_idx = cur_idx;

        for (int next_idx = cur_idx + 1; next_idx < K; ++next_idx) {
            if (out_probs[next_idx * out_probs_stride] <
                out_probs[min_idx * out_probs_stride]) {
                min_idx = next_idx;
            }
        }

        if (min_idx != cur_idx) {
            auto tempValue = out_probs[cur_idx * out_probs_stride];
            out_probs[cur_idx * out_probs_stride] =
                out_probs[min_idx * out_probs_stride];
            out_probs[min_idx * out_probs_stride] = tempValue;

            auto tempIndex = out_indices[cur_idx * out_indices_stride];
            out_indices[cur_idx * out_indices_stride] =
                out_indices[min_idx * out_indices_stride];
            out_indices[min_idx * out_indices_stride] = tempIndex;
        }
    }
}

template <size_t Rank, size_t Axis>
void requeue_descending(float *out_probs, int64_t *out_indices,
                        int64_t out_probs_stride, int64_t out_indices_stride,
                        float candidate_value, int64_t candidate_index,
                        int64_t current_top_index, int64_t K) {
    auto next_candidate_value = out_probs[current_top_index * out_probs_stride];
    auto next_candidata_index =
        out_indices[current_top_index * out_indices_stride];

    out_probs[current_top_index * out_probs_stride] = candidate_value;
    out_indices[current_top_index * out_indices_stride] = candidate_index;

    for (int k = 0; k < K; k++) {
        auto top_value = out_probs[k * out_probs_stride];
        if (next_candidate_value > top_value) {
            requeue_descending<Rank, Axis>(
                out_probs, out_indices, out_probs_stride, out_indices_stride,
                next_candidate_value, next_candidata_index, k, K);
        }
    }
}

template <size_t Rank, size_t Axis>
void requeue_ascending(float *out_probs, int64_t *out_indices,
                       int64_t out_probs_stride, int64_t out_indices_stride,
                       float candidate_value, int64_t candidate_index,
                       int64_t current_down_index, int64_t K) {
    auto next_candidate_value =
        out_probs[current_down_index * out_probs_stride];
    auto next_candidata_index =
        out_indices[current_down_index * out_indices_stride];

    out_probs[current_down_index * out_probs_stride] = candidate_value;
    out_indices[current_down_index * out_indices_stride] = candidate_index;

    for (int k = 0; k < K; k++) {
        auto down_value = out_probs[k * out_probs_stride];
        if (next_candidate_value < down_value) {
            requeue_ascending<Rank, Axis>(
                out_probs, out_indices, out_probs_stride, out_indices_stride,
                next_candidate_value, next_candidata_index, k, K);
        }
    }
}

template <Tensor TInX, Tensor TInK, Tensor TOutProb, Tensor TOutIndice,
          FixedDimension TAxis>
void top_k(const TInX &x, const TInK &k, TOutProb &out_probs,
           TOutIndice &out_indices, TAxis axis, int64_t largest,
           int64_t sorted) {

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
    auto input_strides = x.strides();
    auto out_probes_strides = out_probs.strides();
    auto out_indices_strides = out_indices.strides();
    auto input_p = x.buffer().data();
    auto out_probs_p = out_probs.buffer().data();
    auto out_indices_p = out_indices.buffer().data();
    auto input_stride = x.strides()[axis];
    auto out_probs_stride = out_probs.strides()[axis];
    auto out_indices_stride = out_indices.strides()[axis];
    ntt::apply(
        apply_shape,
        [&](auto, auto input_offset, auto out_probes_offset,
            auto out_indices_offset) {
            auto slice_input_ptr = input_p + input_offset;
            auto slice_probs_ptr = out_probs_p + out_probes_offset;
            auto slice_indices_ptr = out_indices_p + out_indices_offset;

            for (int i = 0; i < K; i++) {
                slice_indices_ptr[i * out_indices_stride] = i;
                slice_probs_ptr[i * out_probs_stride] =
                    slice_input_ptr[i * input_stride];
            }

            for (int i = K; i < inner_size; i++) {
                auto candidate_value = slice_input_ptr[i * input_stride];
                for (int k = 0; k < K; k++) {
                    auto top_value = slice_probs_ptr[k * out_probs_stride];
                    if (largest) {
                        if (candidate_value > top_value) {
                            requeue_descending<rank(), Axis>(
                                slice_probs_ptr, slice_indices_ptr,
                                out_probs_stride, out_indices_stride,
                                candidate_value, i, k, K);
                            break;
                        }
                    } else {
                        if (candidate_value < top_value) {
                            requeue_ascending<rank(), Axis>(
                                slice_probs_ptr, slice_indices_ptr,
                                out_probs_stride, out_indices_stride,
                                candidate_value, i, k, K);
                            break;
                        }
                    }
                }
            }

            if (sorted) {
                if (largest) {
                    sort_descending<rank(), Axis>(
                        slice_probs_ptr, slice_indices_ptr, out_probs_stride,
                        out_indices_stride, K);
                } else {
                    sort_ascending<rank(), Axis>(
                        slice_probs_ptr, slice_indices_ptr, out_probs_stride,
                        out_indices_stride, K);
                }
            }
        },
        input_strides, out_probes_strides, out_indices_strides);
}
} // namespace nncase::ntt
