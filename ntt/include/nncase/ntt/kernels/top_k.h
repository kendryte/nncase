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

template <bool Norm = false, Tensor TInX, Tensor TInK, Tensor TOutProb,
          Tensor TOutIndice, FixedDimension TAxis>
void top_k(const TInX &x, const TInK &k, TOutProb &out_probs,
           TOutIndice &out_indices, TAxis axis, int64_t largest,
           int64_t sorted) {

    using TProbs = element_or_scalar_t<TOutProb>;
    using TIndices = element_or_scalar_t<TOutIndice>;

    constexpr auto rank = TInX::rank();
    constexpr auto Axis = positive_index(dim_t(axis), rank);
    auto K = k(0);
    auto apply_shape = generate_shape<rank>([&](auto i) {
        if (i == Axis)
            return (dim_t)1;
        else
            return (dim_t)x.shape()[i];
    });
    auto inner_size = x.shape()[Axis];
    auto input_strides = x.strides();
    auto out_probes_strides = out_probs.strides();
    auto out_indices_strides = out_indices.strides();
    auto input_p = x.buffer().data();
    auto out_probs_p = out_probs.buffer().data();
    auto out_indices_p = out_indices.buffer().data();
    auto input_stride = x.strides()[Axis];
    auto out_probs_stride = out_probs.strides()[Axis];
    auto out_indices_stride = out_indices.strides()[Axis];
    ntt::apply(
        apply_shape,
        [&](auto, auto input_offset, auto out_probes_offset,
            auto out_indices_offset) {
            auto slice_input_ptr = input_p + input_offset;
            auto slice_probs_ptr = out_probs_p + out_probes_offset;
            auto slice_indices_ptr = out_indices_p + out_indices_offset;
            ntt::u_top_k<TProbs, TIndices, rank, Axis, Norm>(
                inner_size, slice_input_ptr, slice_probs_ptr, slice_indices_ptr,
                input_stride, out_probs_stride, out_indices_stride, K, largest,
                sorted);
        },
        input_strides, out_probes_strides, out_indices_strides);
}
} // namespace nncase::ntt
