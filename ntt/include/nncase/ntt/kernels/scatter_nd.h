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
#include "../utility.h"
#include "copy.h"
#include "nncase/ntt/shape.h"

namespace nncase::ntt {
namespace scatter_nd_detail {
template <Tensor TIn, Tensor TIndex, Tensor TUpdates, Tensor TOut>
void scatter_nd_impl(const TIn &input, const TIndex &indices,
                     const TUpdates &updates, TOut &output) noexcept {
    using TIElem = typename TIn::element_type;
    const auto in_shape = input.shape();
    const auto indices_shape = indices.shape();
    const auto updates_shape = updates.shape();
    const auto out_shape = output.shape();
    const auto in_strides = input.strides();
    const auto indices_strides = indices.strides();
    const auto updates_strides = updates.strides();
    const auto out_strides = output.strides();

    ntt::tensor_copy(input, output);
    constexpr auto k = indices_shape.rank() - dim_one;
    const auto update_indices = indices_shape.template slice<0, k>();
    const auto update_indices_strides = indices_strides.template slice<0, k>();

    const auto in_strides_ = in_strides.template slice<0, indices_shape[k]>();
    const auto updates_strides_ =
        updates_strides.template slice<0, update_indices.rank()>();
    const auto updates_size =
        updates_shape.template slice<update_indices.rank()>().length() *
        fixed_dim_v<sizeof(TIElem)>;

    apply(update_indices, [&](auto idx) {
        auto updates_begin =
            updates.elements().data() + linear_offset(idx, updates_strides_);

        auto data_indices_begin = indices.elements().data() +
                                  linear_offset(idx, update_indices_strides);
        auto data_indices_dim = generate_shape<indices_shape[k]>(
            [&](auto i) { return data_indices_begin[i]; });

        auto data_begin = output.elements().data() +
                          linear_offset(data_indices_dim, in_strides_);

        memcpy(data_begin, updates_begin, updates_size);
    });
}
} // namespace scatter_nd_detail

template <typename TIn, typename TIndex, typename TUpdate, typename TOut>
void scatter_nd(const TIn &input, TIndex &indices, TUpdate &updates,
                TOut &&output) noexcept {
    scatter_nd_detail::scatter_nd_impl(input, indices, updates, output);
}
} // namespace nncase::ntt
