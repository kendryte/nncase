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
#include "shape.h"
#include <cstddef>
#include <cstring>
#include <span>
#include <utility>

namespace nncase::ntt {
namespace utility_detail {
template <size_t OutRank, size_t InRank, size_t... Ints>
inline constexpr ranked_shape<OutRank>
slice_index(const ranked_shape<InRank> &index, const size_t offset,
            std::index_sequence<Ints...>) noexcept {
    return ranked_shape<OutRank>{index[offset + Ints]...};
}

template <size_t OutRank, size_t OffSet = 0, template <size_t...> class A,
          size_t... Dims, size_t... Ints>
inline constexpr auto slice(const A<Dims...> a,
                            std::index_sequence<Ints...>) noexcept {
    return A<a.at(Ints + OffSet)...>{};
}

template <template <size_t...> class T, size_t... ADims, size_t... BDims,
          size_t... I>
inline constexpr bool is_same_seq(const T<ADims...> &a, const T<BDims...> &b,
                                  std::index_sequence<I...>) {
    return ((a[I] == b[I]) && ...);
}
} // namespace utility_detail

template <class U, class T, size_t Extent>
auto span_cast(std::span<T, Extent> span) noexcept {
    using return_type =
        std::conditional_t<Extent == std::dynamic_extent, std::span<U>,
                           std::span<U, Extent * sizeof(T) / sizeof(U)>>;
    return return_type(reinterpret_cast<U *>(span.data()),
                       span.size_bytes() / sizeof(U));
}

template <size_t OutRank, size_t InRank>
inline constexpr ranked_shape<OutRank>
slice_index(const ranked_shape<InRank> &index,
            const size_t offset = 0) noexcept {
    static_assert(OutRank <= InRank, "the out rank must less then inRank");
    return utility_detail::slice_index<OutRank>(
        index, offset, std::make_index_sequence<OutRank>{});
}

template <template <size_t...> class T, size_t... PreDims, size_t... PostDims>
inline constexpr auto concat_fixed_dims(T<PreDims...>,
                                        T<PostDims...>) noexcept {
    return T<PreDims..., PostDims...>{};
}

template <size_t OutRank, size_t OffSet = 0, template <size_t...> class A,
          size_t... Dims>
inline constexpr auto slice_fixed_dims(const A<Dims...> &a) noexcept {
    return utility_detail::slice<OutRank, OffSet>(
        a, std::make_index_sequence<OutRank>{});
}

template <template <size_t...> class T, size_t... ADims, size_t... BDims>
inline constexpr bool is_same_seq(const T<ADims...> &a, const T<BDims...> &b) {
    return sizeof...(ADims) == sizeof...(BDims) &&
           utility_detail::is_same_seq(
               a, b, std::make_index_sequence<sizeof...(ADims)>{});
}

template <typename T>
concept IsFixedTensor = is_fixed_dims_v<typename std::decay_t<T>::shape_type> &&
                        is_fixed_dims_v<typename std::decay_t<T>::strides_type>;

template <typename T>
concept IsRankedTensor =
    is_ranked_dims_v<typename std::decay_t<T>::shape_type> &&
    is_ranked_dims_v<typename std::decay_t<T>::strides_type>;

template <typename T>
concept IsFixedDims = is_fixed_dims_v<T>;

template <class T> struct is_vector : std::false_type {};

template <template <typename, size_t...> class V, typename T, size_t... Lanes>
struct is_vector<V<T, Lanes...>> : std::true_type {};

template <class T> inline constexpr bool is_vector_v = is_vector<T>::value;

} // namespace nncase::ntt
