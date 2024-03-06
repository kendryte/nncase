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
#include "../shape.h"

namespace nncase::ntt::shape_infer {
namespace detail {

template <size_t Lanes, size_t Axis, size_t Rank>
static constexpr size_t
unpacked_index_by_shape_dim(const ranked_shape<Rank> &input_index,
                            const size_t i) {
    if (i == Axis) {
        return input_index[i] * Lanes;
    }

    return input_index[i];
}

template <size_t Lanes, size_t Axis, size_t Rank, size_t... I>
static constexpr auto
unpacked_index_by_shape_impl(const ranked_shape<Rank> &input_index,
                             std::index_sequence<I...>) {
    {
        return ranked_shape<Rank>{
            unpacked_index_by_shape_dim<Lanes, Axis>(input_index, I)...};
    }
}

} // namespace detail
template <size_t Lanes, size_t Axis, size_t Rank>
static constexpr auto unpacked_index_by_shape(const ranked_shape<Rank> &input_index) {
    return detail::unpacked_index_by_shape_impl<Lanes, Axis>(
        input_index, std::make_index_sequence<Rank>{});
}

} // namespace nncase::ntt::shape_infer