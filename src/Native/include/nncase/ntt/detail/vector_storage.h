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
#include "../shape.h"
#include "../utility.h"
#include <vector>

namespace nncase::ntt {
template <class T, size_t... Lanes> class basic_vector;
template <class T, size_t... Lanes> struct vector_storage_traits;

namespace detail {
template <class TTraits, class TIndex> class vector_storage_element_proxy {
  public:
    using buffer_type = typename TTraits::buffer_type;
    using element_type = typename TTraits::element_type;

    constexpr vector_storage_element_proxy(buffer_type &buffer,
                                           TIndex index) noexcept
        : buffer_(buffer), index_(index) {}

    constexpr operator element_type() const noexcept {
        return TTraits::get_element(buffer_, index_);
    }
    constexpr vector_storage_element_proxy &
    operator=(element_type value) noexcept {
        TTraits::set_element(buffer_, index_, value);
        return *this;
    }

  private:
    buffer_type &buffer_;
    TIndex index_;
};
} // namespace detail

template <class T, size_t Lane> struct vector_storage_traits<T, Lane> {
    using buffer_type = std::array<T, Lane>;
    using element_type = T;

    static constexpr T &element_at(std::array<T, Lane> &array,
                                   ranked_shape<1> index) noexcept {
        return array[index[0]];
    }

    static constexpr const T &element_at(const std::array<T, Lane> &array,
                                         ranked_shape<1> index) noexcept {
        return array[index[0]];
    }
};

template <class T, size_t OuterLane, size_t... InnerLanes>
struct vector_storage_traits<T, OuterLane, InnerLanes...> {
    using buffer_type = std::array<basic_vector<T, InnerLanes...>, OuterLane>;
    using element_type = T;

    template <size_t IndexRank>
    static constexpr decltype(auto)
    element_at(buffer_type &vec, ranked_shape<IndexRank> index) noexcept {
        auto &inner_vector = vec[index[0]];
        auto remaining_index = slice_index<IndexRank - 1>(index, 1);
        if constexpr (IndexRank > 1) {
            return inner_vector(remaining_index);
        } else {
            return inner_vector;
        }
    }

    template <size_t IndexRank>
    static constexpr decltype(auto)
    element_at(const buffer_type &vec, ranked_shape<IndexRank> index) noexcept {
        auto &inner_vector = vec[index[0]];
        auto remaining_index = slice_index<IndexRank - 1>(index, 1);
        if constexpr (IndexRank > 1) {
            return inner_vector(remaining_index);
        } else {
            return inner_vector;
        }
    }
};
} // namespace nncase::ntt
