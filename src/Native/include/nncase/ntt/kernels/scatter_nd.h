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
#include "copy.h"
#include "../utility.h"

namespace nncase::ntt {

namespace scatter_nd_detail {

template <IsFixedTensor TIn, IsFixedTensor TIndex, IsFixedTensor TUpdates,
          IsFixedTensor TOut>
void scatter_nd_impl(const TIn &input, const TIndex &indices,
                     const TUpdates &updates, TOut &&output) noexcept {
    using TIElem = typename TIn::element_type;
    [[maybe_unused]] constexpr auto in_shape = typename TIn::shape_type{};
    constexpr auto indices_shape = typename TIndex::shape_type{};
    constexpr auto updates_shape = typename TUpdates::shape_type{};
    [[maybe_unused]] constexpr auto out_shape = typename std::decay_t<TOut>::shape_type{};
    constexpr auto in_strides = typename TIn::strides_type();
    constexpr auto indices_strides = typename TIndex::strides_type();
    constexpr auto updates_strides = typename TUpdates::strides_type();
    [[maybe_unused]] constexpr auto out_strides = typename std::decay_t<TOut>::strides_type{};

    ntt::tensor_copy(input, output);
    constexpr auto k = indices_shape.rank() - 1;
    auto update_indices = slice_fixed_dims<k>(indices_shape);
    auto update_indices_strides = slice_fixed_dims<k>(indices_strides);

    auto in_strides_ = slice_fixed_dims<indices_shape.at(k)>(in_strides);
    
    auto updates_strides_ = slice_fixed_dims<update_indices.rank()>(updates_strides);
    auto updates_size = sizeof(TIElem);
    for (auto i = update_indices.rank(); i < updates_shape.rank(); ++i) {
        updates_size *= updates_shape.at(i);
    }

    static_assert(IsScalar<typename std::decay_t<TOut>::element_type>,
                  "Only support scalar type for now");

    apply(
        update_indices, [&](auto idx) {
            auto updates_begin = updates.elements().data() + linear_offset(idx, updates_strides_);

            auto data_indices_begin = indices.elements().data() + linear_offset(idx, update_indices_strides);
            ranked_shape<indices_shape.at(k)> data_indices_dim;
            for (auto i = 0; i < indices_shape.at(k); ++i) {
                data_indices_dim.at(i) = *(data_indices_begin + i);
            }

            auto data_begin = output.elements().data() + linear_offset(data_indices_dim, in_strides_);

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
